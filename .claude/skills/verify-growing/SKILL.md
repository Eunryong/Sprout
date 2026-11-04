---
name: verify-growing
description: Drive the Growing Agent web game (a tiny transformer "child" the player raises from 3 to 7 years by choosing say / show / leave each scene) the way a player does, and capture proof. Use when a change touches server/web, server/game, the engine, the Narrator/Director or the page, or when asked to "verify", "check in the browser", or "prove it works" for this repo.
---

# Verify Growing (web game, mock LLM)

Primary surface: the web page served by `python -m server.web` (FastAPI + one static page `server/web/static/index.html`).
Secondary: the terminal game `python -m server.cli play`. Research scripts under `scripts/` are experiments, not a user surface.

Read `features/README.md` first, then the feature file that matches the change. Every command below is run from the repo root.

> Status: written from the source (routes in `server/web/app.py`, ids in `index.html`). It has **not** been run end to end yet. Treat the first real run as the proof of this skill and fix any step that does not match, then record the fix here.

## Launch

```bash
.claude/skills/verify-growing/scripts/launch.sh 1     # prints: RUN_DIR=... PORT=... URL=...
```

- Uses `GROWING_LLM=mock` (no network, no API spend; it overrides the live key in `.env`), `--lang en`, a free port, and a fresh save in `saves/verify/<run-id>/save`. The save is a new child **starting at 36 months** (3 years) in the default 0-7 years world.
- Never use `saves/web/current`, `saves/web/check`, or port 8000: those are the user's own game and `.claude/launch.json`'s shared preview.
- Ready means the first scene has at least 2 options and a narration (the script polls `/api/scene`), not only that the port answers. With the mock it takes a few seconds; live Narrator/Director would take 10-25 s, but verification does not use live.
- Keep `RUN_DIR` and `PORT`; every other step needs them.
- The web server runs in two-step mode by default: step 1 is the parent's part (show or say on an option, or leave), step 2 is the child's solo moment. `GROWING_TWO_STEP=0` gives single-step scenes, but the player never sees that, so do not verify in it unless the change is about it.

## Doctor

```bash
.claude/skills/verify-growing/scripts/doctor.sh "$RUN_DIR"
```

Read-only. Passes only if our recorded pid is alive and owns the port, `/api/info` says `mode=mock` and `lang=en`, and the save is inside `RUN_DIR`. Run it before driving and whenever anything looks off (empty page, 409/500 from the API, a scene that never comes). If it fails, do not drive; clean up and relaunch. Never drive an instance this run did not start.

## Drive

Two harnesses, use both when the change is user-visible:

**1. Browser** (the built-in browser pane, `mcp__Claude_Browser__*`, or Claude in Chrome). Open `$URL/` (add `?research=1` only for the research view). Stable handles from `index.html`:

| What | Handle |
|---|---|
| scene counter / age / stage | `#sceneNo`, `#stageChip`, `#stepChip`, `#banner` (stage intro) |
| scene text | `#story` |
| option cards | `#opts li.opt`, title in `h3` |
| parent acts on an option | `#opts li.opt button[data-part="say"]` or `[data-part="show"]` (label text is localised and the same for both, address by `data-part` and the card's `h3`) |
| let the child do it alone | `#leave` (step 1) and the card button `[data-part="leave"]` inside `li.auto-opt` / `li.solo-opt` |
| what came of it | `#result` (children: `.who`, `.child-said`, plain narration, `.world`, `.learned`) |
| continue | `#next` (inside `#after`) |
| end of life | `#end`, `#life` |
| language | `#langBtn` |
| new child | `#newGame` (opens `#adopt` with temperament buttons) |
| talk to the child | `#openQaBtn`, `#qaInput`, `#qaSendBtn`, `#qaMessages`, `#qaCloseBtn` |
| error banner | `#err` (role=alert), must stay hidden in a good run |

Wait for `#busy` to be hidden and for the section you expect (`#choose` or `#after`) instead of sleeping.

**2. HTTP** (for logic and side effects; the page uses the same calls). `PORT` from launch:

```bash
B=http://127.0.0.1:$PORT
curl -s $B/api/info                                      # lived, ended, mode, lang, world
curl -s $B/api/scene                                     # number, narration, options[{index,text,expects,memory,lean,seen,pull}], step, can_act, age_months
curl -s -X POST $B/api/play -H 'content-type: application/json' -d '{"part":"show","option":0}'   # part: say | show | leave
curl -s $B/api/summary                                   # only meaningful once the child is grown
curl -s $B/api/children                                  # [] here (empty children dir)
```

The API returns `result` (solved/partial/nothing/worse), `expects` and other research numbers on purpose. The rule "no grading" is about what the **page shows a player**, so judge it on the rendered page, not on the JSON.

Terminal game (when the change is in `server/cli.py` or `server/game/simple.py`): run it in an isolated tmux session with `GROWING_LLM=mock uv run python -m server.cli play --seed 1 --save "$RUN_DIR/cli-save"`, send `show 0` / `say 0` / `leave` / `quit`, and capture the pane.

## Evidence

Put everything in `$RUN_DIR/evidence/<feature-id>/` (it survives cleanup). Proof standard:

- Exercise the real player path: click the buttons in the page, or POST exactly what the page POSTs. No internal setters, no editing the save by hand, no test-only endpoints.
- Capture the action and the state it caused: before-scene (`/api/scene`), the click or request, the result card text, the next scene. A final screenshot alone is not proof.
- Check the side effect in a second view: `/api/info` `lived` went up by one, `$RUN_DIR/save/` files changed (`meta.json`, event log), the next `/api/scene` `number` is one higher.
- Screenshots show the page identity (title "Sprout", `#sceneNo`). Save the rendered `#result` text with `read_page`/`get_page_text` beside the screenshot.
- Record the feature id and entry point in each artifact name, e.g. `scene-play-show-opt0.png`.
- Mock prose is deterministic filler; it proves wiring and guards, not prose quality. Say so in the report and do not claim anything about live Director or Narrator behaviour.
- Report an unreachable path with the command tried and the unmet precondition. Do not call a path verified because a different path worked.

## Cleanup

```bash
.claude/skills/verify-growing/scripts/cleanup.sh "$RUN_DIR"
ls "$RUN_DIR/evidence"        # confirm the proof is still there
```

Stops only the pids `launch.sh` recorded, removes `save/`, `save-old-*`, `children/`, keeps `evidence/` and `server.log`. Run it after every failed attempt too. It refuses any directory outside `saves/verify/`. If a step started a second server (the New child flow can relaunch the process, see the feature file), cleanup handles it only through the port this run owns; check `lsof -ti tcp:$PORT -sTCP:LISTEN` is empty afterwards.

## Helpers

| Script | Invocation | Does |
|---|---|---|
| `scripts/launch.sh [seed]` | `.claude/skills/verify-growing/scripts/launch.sh 1` | isolated mock server, prints `RUN_DIR`, `PORT`, `URL` |
| `scripts/doctor.sh <RUN_DIR>` | `.../doctor.sh "$RUN_DIR"` | read-only health check, exit 0 = drive it |
| `scripts/cleanup.sh <RUN_DIR>` | `.../cleanup.sh "$RUN_DIR"` | stop our server, delete scratch, keep evidence |

## Do not

- Do not run the live LLM (`GROWING_LLM=live`) in verification: it spends the user's API key and takes minutes. A live check is a separate, explicit request.
- Do not run the research scripts (`scripts/exp_*.py`) as UI verification; they take minutes and test the model, not the page. For model or engine changes the proof is the matching single test file (`uv run pytest tests/test_<area>.py`) plus one driven scene.
- Do not weaken a failing journey to make it pass; report the failure and put the cause in the feature's Gotchas.
- Do not print or copy anything from `.env`.
