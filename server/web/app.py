"""The web screen of the simple-choice game (docs/DIRECTION.md M3): a thin JSON layer over SimpleGame and one static page.

  GET  /api/info     the game in one line: scenes lived, seed, world premise, who writes the prose (mock / live)
  GET  /api/scene    the scene now: narration, options with the child's own expectations
  POST /api/play     {"part": "say" | "show" | "leave", "option": n}  ->  what came of it
  POST /api/new      {"seed": n}  start again; the old game is kept next to it, never deleted

One game per server, one request at a time (the child learns after every scene; a lock keeps the steps in order).
"""
import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from server import i18n
from server.agents.director import ProposalError
from server.agents.factory import load_llm_cfg
from server.game.memory import probe_child_brain
from server.game.simple import Part, SimpleGame
from server.world.events import RuleError
from server.world.ontology import ONTOLOGY_HASH, STAGES, age_months, stage_of, voice_of

STATIC = Path(__file__).parent / "static"


class PlayIn(BaseModel):
    part: Part | None = None
    option: int | None = None
    option_index: int = 0
    choice: int | None = None
    custom_text: str | None = None


class AskIn(BaseModel):
    question: str


class NewIn(BaseModel):
    seed: int | None = None
    child: str | None = None
    persona: str | None = None
    name: str | None = None


class LangIn(BaseModel):
    lang: str


NAME_POOL = ["bori", "kong", "byeol", "narae", "haneul", "bada", "dami", "sol"]  # cute defaults, same style as /api/raise


def _validated_name(raw: str | None) -> str | None:
    """Strip a requested name; None when missing/empty, 422 when longer than 20 chars."""
    if raw is None or not raw.strip():
        return None
    name = raw.strip()
    if len(name) > 20:
        raise HTTPException(422, "name must be 1-20 characters")
    return name


def resolve_child_name(raw: str | None, seed: int) -> str:
    """Strip and validate a requested name; missing/empty falls back to a seed-picked default."""
    return _validated_name(raw) or NAME_POOL[seed % len(NAME_POOL)]


def _read_child_name(d: Path) -> str | None:
    try:
        v = json.loads((d / "meta.json").read_text()).get("child_name")
        return v if isinstance(v, str) and v else None
    except (OSError, ValueError):
        return None


def _rotation_suffix() -> str:
    """Microsecond-precision save-rotation suffix: two /api/new calls in the same second must not collide."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _write_child_name(d: Path, name: str) -> None:
    p = d / "meta.json"
    try:
        meta = json.loads(p.read_text())
    except (OSError, ValueError):
        return
    meta["child_name"] = name
    p.write_text(json.dumps(meta, indent=1), encoding="utf-8")


class Holder:
    """The running game and where it is saved."""

    def __init__(self, save_dir: Path, seed: int, two_step: bool | None = None):
        if two_step is None:
            two_step = os.environ.get("GROWING_TWO_STEP", "0" if "pytest" in sys.modules else "1") == "1"
        self.two_step = two_step
        self.dir, self.seed, self.lock = Path(save_dir), seed, threading.Lock()
        has_3yo_stage = any(s.get("from", 0) >= 36 for s in STAGES)
        if (self.dir / "meta.json").exists():
            if (self.dir / "raised.json").exists():
                self.restart(seed)
                return
            try:
                self.game = SimpleGame.load(self.dir, two_step=self.two_step)
                if has_3yo_stage and self.game.age_now() < 36:
                    self.game.bootstrap_age_three()
            except ValueError:
                shutil.rmtree(self.dir, ignore_errors=True)
                self.game = SimpleGame.new(self.dir, seed=seed, two_step=self.two_step, start_at_three=has_3yo_stage)
        else:
            self.game = SimpleGame.new(self.dir, seed=seed, two_step=self.two_step, start_at_three=has_3yo_stage)

    def adopt(self, src: Path, name: str | None = None) -> None:
        """Take over a child that was raised to three: its save is copied here; the earlier child is kept aside."""
        _validated_name(name)  # fail before anything is moved aside
        if (self.dir / "meta.json").exists() or self.dir.exists():
            self.dir.rename(self.dir.with_name(f"{self.dir.name}-old-{_rotation_suffix()}"))
        shutil.copytree(src, self.dir)
        self.game = SimpleGame.load(self.dir, two_step=self.two_step)
        self.seed = self.game.s.seed
        _write_child_name(self.dir, resolve_child_name(name, self.seed))

    def restart(self, seed: int, persona: str | None = None, name: str | None = None) -> None:
        child_name = resolve_child_name(name, seed)  # fail before anything is moved aside
        if (self.dir / "meta.json").exists():
            old = self.dir.with_name(f"{self.dir.name}-old-{_rotation_suffix()}")
            self.dir.rename(old)
            for fname in ("ontology.yaml", "world.json"):  # the running world may live in the save itself: the next child grows up in the same one
                if (old / fname).exists():
                    self.dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(old / fname, self.dir / fname)
        self.seed = seed
        has_3yo_stage = any(s.get("from", 0) >= 36 for s in STAGES) or (persona is not None)
        self.game = SimpleGame.new(self.dir, seed=seed, two_step=self.two_step, start_at_three=has_3yo_stage, persona=persona or "warm")
        _write_child_name(self.dir, child_name)


def premise_text(sheet) -> str:
    if sheet is None:
        return ""
    return "저마다 다른 집, 저마다 다른 세계예요. 이 집에서는 물건들이 이 집만의 방식으로 쓰여요." if i18n.is_ko() and sheet.source == "template" else sheet.premise


def _load_qa_history(save_dir: Path) -> list[dict]:
    p = save_dir / "qa_history.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_qa_history(save_dir: Path, history: list[dict]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    p = save_dir / "qa_history.json"
    p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def child_world(d: Path) -> str | None:
    """The ontology hash a saved child grew up in."""
    try:
        return json.loads((d / "meta.json").read_text()).get("ontology")
    except (OSError, ValueError):
        return None


def raised_info(d: Path) -> dict | None:
    p = d / "raised.json"
    return {"name": d.name} | json.loads(p.read_text()) if p.exists() else None  # the name it was raised under, else its folder


def create_app(save_dir: str | Path = "saves/web/current", seed: int = 0, prefetch: bool = True, children_dir: str | Path | None = None, two_step: bool | None = None) -> FastAPI:
    """`prefetch`: while the player reads a result, the next scene is already being made (the Director and the Narrator take 10-25 seconds live)."""
    app = FastAPI(title="Sprout", docs_url=None, redoc_url=None)
    h = Holder(Path(save_dir), seed, two_step=two_step)
    app.state.holder, app.state.prefetch = h, None
    kids = Path(children_dir) if children_dir else None

    def warm() -> None:
        with h.lock:
            try:
                h.game.scene()
            except Exception:  # noqa: BLE001  the page asks again and shows the error itself
                pass

    def prefetch_next() -> None:
        if prefetch:
            app.state.prefetch = threading.Thread(target=warm, daemon=True)
            app.state.prefetch.start()

    prefetch_next()  # the first scene while the page loads

    @app.get("/")
    def index():
        return HTMLResponse((STATIC / "index.html").read_text().replace("__LANG__", i18n.lang()))  # the page starts in the language the server speaks

    @app.post("/api/lang")
    def set_lang(body: LangIn):
        """The player's language (en | ko). Only the words change: the child is the same child. Waits for a scene in the making."""
        if body.lang not in i18n.LANGS:
            raise HTTPException(422, f"lang must be one of {i18n.LANGS}")
        with h.lock:
            i18n.set_lang(body.lang)
        return {"lang": i18n.lang()}

    @app.get("/api/info")
    def info():
        with h.lock:
            g = h.game
            sheet = g.s.world_sheet()
            d, n = g.s.director, g.s.narrator.stats.summary()
            raised = None
            return {"raised": raised, "lang": i18n.lang(), "lived": g.lived, "seed": h.seed, "child_name": _read_child_name(h.dir), "premise": premise_text(sheet),
                    "world": sheet.source if sheet else f"hand-written ({'0-7 years' if STAGES else 'small'})", "curiosity": g.s.game.curiosity,
                    "mode": os.environ.get("GROWING_LLM") or load_llm_cfg().mode, "ended": g.ended,
                    "director": {"source": d.last_source if d.stats.calls + d.stats.no_client else None, "error": d.last_error},  # who wrote today's scenes: director:<model> or template
                    "narrator": {"calls": n["calls"], "ok": n["ok"], "fallback": n["fallback"]},
                    "qa_history": _load_qa_history(h.dir)}

    @app.get("/api/scene")
    def scene():
        with h.lock:
            try:
                return h.game.scene().model_dump()
            except HTTPException:
                raise
            except (RuleError, ProposalError) as e:
                raise HTTPException(409, str(e))
            except Exception as e:  # noqa: BLE001  a rejected draw is a conflict, never a raw 500
                raise HTTPException(409, str(e))

    @app.get("/api/summary")
    def summary():
        with h.lock:
            return h.game.summary().model_dump()

    @app.post("/api/play")
    def play(body: PlayIn):
        with h.lock:
            try:
                h.game.scene()
                part = body.part
                if part is None:
                    part = "show" if not h.game.can_act() else "say"
                if body.choice is not None:
                    opt = body.choice
                elif "option" in body.model_fields_set:
                    opt = body.option
                elif body.part is not None and body.part != "leave":
                    opt = body.option
                else:
                    opt = body.option_index

                sig = inspect.signature(h.game.play)
                kwargs = {}
                if "custom_text" in sig.parameters and body.custom_text is not None:
                    kwargs["custom_text"] = body.custom_text

                if "option_index" in sig.parameters:
                    try:
                        turn = h.game.play(opt, **kwargs)
                    except TypeError:
                        turn = h.game.play(opt)
                else:
                    try:
                        turn = h.game.play(part, opt, **kwargs)
                    except TypeError:
                        turn = h.game.play(part, opt)
                turn_data = turn.model_dump() if hasattr(turn, "model_dump") else turn
                if isinstance(turn_data, dict):
                    turn_data.setdefault("outcome_text", turn_data.get("narration", ""))
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(400, str(e))
            except (RuleError, ProposalError) as e:
                raise HTTPException(409, str(e))
            except Exception as e:  # noqa: BLE001  a rejected draw is a conflict, never a raw 500
                raise HTTPException(409, str(e))
        prefetch_next()  # after the lock is released: the next scene is made while the player reads this result
        return turn_data

    @app.get("/api/ask")
    def api_ask_get():
        with h.lock:
            return {"history": _load_qa_history(h.dir)}

    @app.post("/api/ask")
    def api_ask(body: AskIn):
        req = body
        with h.lock:
            g = h.game
            summary_data = g.summary()
            _lang = i18n.lang() if hasattr(i18n, "lang") else ("ko" if i18n.is_ko() else "en")

            # Age
            age = getattr(g.s, "age", None)
            if age is None:
                age = getattr(summary_data, "age", f"{g.age_now()} months")

            v = g.s.view() if hasattr(g.s, "view") and callable(g.s.view) else None
            age_months_val = getattr(v, "age_months", None) if v is not None else getattr(g.s, "age_months", None)
            if age_months_val is None:
                if hasattr(g, "age_now") and callable(g.age_now):
                    age_months_val = g.age_now()
                elif hasattr(g.s, "state") and hasattr(g.s.state, "day"):
                    age_months_val = age_months(g.s.state.day)
                else:
                    age_months_val = 36
            if v is None or not hasattr(v, "age_months"):
                v = type("SceneViewObj", (), {"age_months": age_months_val})()

            # Stage
            stage_title = ""
            if hasattr(g.s, "stage") and callable(g.s.stage):
                stg_obj = g.s.stage()
                stage_title = getattr(stg_obj, "title", getattr(stg_obj, "name", str(stg_obj)))
            elif hasattr(g.s, "stage"):
                stage_title = getattr(g.s.stage, "title", getattr(g.s.stage, "name", str(g.s.stage)))
            else:
                stg_info = stage_of(age_months_val) if callable(stage_of) else None
                stg_dict = i18n.stage(stg_info) if stg_info else None
                stage_title = stg_dict.get("title", stg_dict.get("name", "")) if stg_dict else ""

            # Scenes
            scenes = getattr(g.s, "scenes_played", None)
            if scenes is None:
                scenes = getattr(g, "lived", getattr(summary_data, "scenes", 0))

            # Neural network brain state & parenting relationship
            if hasattr(g, "knowledge") and getattr(g, "knowledge", None) is not None:
                g.s.knowledge = g.knowledge
            brain = probe_child_brain(g.s, lang=_lang)
            brain.setdefault("model_beliefs", [])
            brain.setdefault("learned_habits", [])
            brain.setdefault("trial_memories", [])
            brain.setdefault("unlearned_note", "")
            brain.setdefault("parent_relationship", {})

            learned_habits = brain["learned_habits"]
            trial_memories = brain["trial_memories"]
            learned_count = len(learned_habits)
            moments = [b["summary"] for b in brain["model_beliefs"][-5:]] if brain["model_beliefs"] else []

            cur_narration = getattr(g, "_current_solo_narration", "") or getattr(g, "_current_scene_narration", "")
            cur_options = getattr(g, "_current_solo_option_texts", []) or getattr(g, "_current_option_texts", [])

            history = _load_qa_history(h.dir)

            ask_ctx = {
                **brain,
                "brain_state": brain,
                "age": age,
                "age_months": v.age_months,
                "stage": stage_title,
                "moments": moments,
                "current_scene": {"narration": cur_narration, "options": cur_options},
                "dialogue_history": history[-8:],
            }

            try:
                answer = g.s.narrator.ask_child(req.question, ask_ctx, age_months=v.age_months, lang=_lang)
            except TypeError:
                answer = g.s.narrator.ask_child(req.question, ask_ctx)

            history.append({
                "question": req.question,
                "answer": answer,
                "age": age,
                "stage": stage_title,
                "scenes": scenes,
            })
            _save_qa_history(h.dir, history)

            return {
                "ok": True,
                "question": req.question,
                "answer": answer,
                "age": age,
                "age_months": age_months_val,
                "stage": stage_title,
                "scenes": scenes,
                "learned_count": learned_count,
                "learned_habits": learned_habits,
                "trial_memories": trial_memories,
                "moments": moments[-6:],
                "brain_state": brain,
                "parent_relationship": brain["parent_relationship"],
                "history": history,
            }

    @app.get("/api/children")
    def children():
        """The children raised to three that wait to be taken over (muse-spark was their parent for the first years). `other_world`: the child grew up
        in a different world from the running one, so taking it over restarts the game in its world."""
        found = [raised_info(d) | {"name": d.name, "other_world": child_world(d) != ONTOLOGY_HASH} for d in sorted(kids.iterdir()) if d.is_dir() and raised_info(d)] if kids and kids.exists() else []
        return found

    def relaunch(*flags: str) -> None:
        """The world is read once, when the program starts: another world means the same game started again, with the same port and language."""
        keep, skip, argv = [], False, sys.argv[1:]
        for i, x in enumerate(argv):
            if skip:
                skip = False
            elif x in ("--child", "--world", "--seed", "--save", "--lang"):
                skip = True
            elif x not in ("--fresh", "--new-world", "--small"):
                keep.append(x)
        args = ["-m", "server.web", "--save", str(h.dir), "--lang", i18n.lang(), *keep, *flags]
        threading.Timer(0.8, lambda: os.execv(sys.executable, [sys.executable, *args])).start()

    @app.post("/api/new")
    def new(body: NewIn):
        with h.lock:
            try:
                if body.persona:
                    h.restart(h.seed + 1 if body.seed is None else body.seed, persona=body.persona, name=body.name)
                elif body.child and body.child in ("warm", "playful", "busy", "strict") and not (kids and (kids / body.child / "raised.json").exists()):
                    h.restart(h.seed + 1 if body.seed is None else body.seed, persona=body.child, name=body.name)
                elif body.child:
                    src = (kids / body.child) if kids else None
                    if src is None or "/" in body.child or not (src / "raised.json").exists():
                        raise HTTPException(404, f"no raised child called {body.child!r}")
                    if child_world(src) != ONTOLOGY_HASH:
                        relaunch("--child", str(src))
                        return {"restarting": True}
                    try:
                        h.adopt(src, name=body.name)
                    except ValueError as e:
                        raise HTTPException(409, str(e))
                else:
                    h.restart(h.seed + 1 if body.seed is None else body.seed, name=body.name)
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001  a rotation clash is a conflict, never a raw 500
                raise HTTPException(409, str(e))
        prefetch_next()
        return {"seed": h.seed}

    raising: dict = {}  # the new child being raised to three in the background: {name, proc, started}

    @app.post("/api/raise")
    def raise_new():
        """A new child in a new randomly drawn world: muse-spark raises it from birth to three in the background (a few minutes live), then it waits
        in the chooser like the others and the page takes it over."""
        from server.agents.parent import PERSONAS
        if kids is None:
            raise HTTPException(404, "no place for raised children")
        with h.lock:
            if raising and raising["proc"].poll() is None:
                return {"name": raising["name"]}  # one at a time
            ws = int(time.time()) % 100000
            names = ["bori", "kong", "byeol", "narae", "haneul", "bada", "dami", "sol"]
            name = f"{names[ws % len(names)]}{ws}"
            kids.mkdir(parents=True, exist_ok=True)
            logs = kids.parent / "children-raising"
            logs.mkdir(parents=True, exist_ok=True)
            personas = sorted(PERSONAS)
            cmd = [sys.executable, "-m", "server.game.raise_child", "--seed", str(ws), "--persona", personas[ws % len(personas)], "--world-seed", str(ws), "--out", str(kids / name)]
            if os.environ.get("GROWING_LLM") == "mock":
                cmd.append("--mock")
            raising.clear()
            raising.update(name=name, started=time.time(), proc=subprocess.Popen(cmd, stdout=(logs / f"{name}.log").open("w"), stderr=subprocess.STDOUT))  # it draws its own world, whatever this server runs in
        return {"name": name}

    @app.get("/api/raising")
    def raising_state():
        if not raising:
            return {"state": "none"}
        code = raising["proc"].poll()
        done = code == 0 and (kids / raising["name"] / "raised.json").exists()
        return {"state": "running" if code is None else "done" if done else "failed", "name": raising["name"], "seconds": int(time.time() - raising["started"])}

    return app


app = create_app()

