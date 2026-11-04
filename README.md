# Sprout — raise a tiny transformer by choosing what to do

A randomly initialised ~110k-parameter transformer ("the child") learns for real from the choices you make.
A Director (a Meta model, later) invents situations and options; the engine decides what each option does; you pick
one; the child learns from your pick immediately. **Current direction: [docs/DIRECTION.md](docs/DIRECTION.md)** (it wins over the
older [docs/DESIGN.md](docs/DESIGN.md) and [docs/WORLD.md](docs/WORLD.md), which are background and experiment logs).

```bash
uv sync
uv run pytest                                           # ~30 s
uv run python -m server.cli play --seed 1               # simple choice: one scene at a time, say / show / leave (M1)
uv run python -m server.web --seed 1                    # the 0-7 years game as a web page, http://127.0.0.1:8000 (M3); resumes saves/web/current. --small = the five-problem world
uv run python -m server.web --new-world --seed 31       # the Director (Meta API, GROWING_LLM=live in .env) makes the world first; takes a minute or two
uv run python -m server.web --world saves/worlds/fcf58d76bf3d/ontology.yaml   # a world the Director already made
uv run python -m server.cli new --seed 1                # play in the terminal (type `help`)
uv run python -m server.cli autoplay --seed 1 --days 3  # synthetic player, prints the evening quizzes
uv run python scripts/exp_llm.py --client sloppy --seeds 0-9       # how often LLM answers are usable (fault-injected stand-in)
uv run python scripts/exp_ea.py --seeds 200-239 --jobs 12          # E-A: how early parenting styles shape the child (WORLD.md 19)
uv run python scripts/exp_ea_engine.py --seeds 200-239 --jobs 12   # the same, as engine events through Session (WORLD.md 20)
uv run python scripts/exp_obey.py --cells chooser commander:0.25 commander:0.5 --out scripts/out/obey_default.json   # when blind obedience appears (WORLD.md 21)
GROWING_ONTOLOGY=$PWD/config/experiments/onto_age03.yaml GROWING_LLM=mock uv run python scripts/exp_gate.py --seeds 200-223 --days 8 --grid "age03 pd8|per_day=8"   # the 0-3 years world (WORLD.md 22)
# with the age schedule (WORLD.md 22.7): same command; add GROWING_NO_SCHEDULE=1 for the all-open control, --cond none for learning off
GROWING_TONE_INPUT=1 uv run python scripts/exp_tone_hear.py --out scripts/out/tone_hear.json   # the child hears the tone (WORLD.md 22.9); exp_tone.py = tone changes only how much is learned (22.8)
uv run python scripts/exp_autonomy.py --out scripts/out/autonomy.json   # own experience learned more, replay that fades (WORLD.md 22.10); exp_oneshot.py = how much one exposure leaves
uv run python scripts/exp_outcome.py --out scripts/out/outcome.json   # a wrong choice learned by what it did (WORLD.md 22.11; needs the -1 world, config/experiments/onto_age03_harm.yaml, set by the script)
uv run python scripts/exp_burn.py --out scripts/out/burn.json   # being burned by the parent's word, own burn, regret (WORLD.md 22.12; --arms base worse1 worse3 both own all all+regret)
uv run python -m server.world.worldgen --seed 7 --out saves/worlds/w7 --check   # a generated world as an ontology file (WORLD.md 22.13); play it with GROWING_ONTOLOGY=saves/worlds/w7/ontology.yaml
GROWING_LLM=mock uv run python -m server.world.worldgen --seed 3 --out saves/worlds/w3 --director --mood "rainy" --check   # the Director proposes the world (live: set the key in .env.local)
uv run python scripts/exp_worlds.py --worlds 0-19 --out scripts/out/worlds.json   # are generated worlds learnable? (acceptance gate)
uv run python scripts/trace_child.py --scenes 12 --parent mixed   # watch one child scene by scene: the parent's part, the child's expectations, what it did, what came of it (WORLD.md 22.14)
uv run python scripts/exp_worlds.py --worlds 0-19 --out scripts/out/worlds_cons.json   # default child learns what happens, not which option is right (WORLD.md 22.14)
uv run python scripts/exp_gate.py --seeds 200-239 --days 10 --grid "pd8|per_day=8"   # multi-seed gate
```

## Layout

| Path | What |
|---|---|
| `config/ontology.yaml` | the closed ontology + rules table: the only source of "which option is good" |
| `config/{game,model}.yaml` | day length, quiz sizes, model size, online-learning settings |
| `server/world/` | situations/options, rules in action, template generator, event-sourced engine |
| `server/learning/` | the child: tokens, tiny transformer, online learner, held-out quiz, checkpoints |
| `server/agents/` | Director (proposes, repairs, falls back), Narrator (LLM prose behind text guards, fixed-phrase fallback), LLM client chain with mock |
| `server/game/` | `Session` (the game loop), saves, synthetic player, synthetic parents (`parents.py`) |
| `scripts/` | experiments (`exp_gate.py` = current gate), `legacy_v1/` = v1 experiments |
| `archive/` | snapshot of the v1 skeleton |

## Honesty rules (enforced by tests)

- The child's input has tags, a name token and each option's verb + target *attributes*. Never a kind, a value, an
  outcome or any narration.
- Only the engine's rules table says which option is best. Training code does not import it (AST test).
- The quiz uses scenes the engine refuses to offer; baselines: random, the immutable untrained `ckpt_000`.
- The parent can `talk` (name a thing), `tell` the child which option to take, and let it choose first; each costs attention
  (8 units a day). How good an instruction was and how the child reacted are computed by the engine from the rules table
  and kept in the event log; the child only ever sees the told option's verb and attributes (`CMD`).
- A Director proposal is validated by the engine; one repair try, then the template: the game goes on.
- Player-facing prose (teasers, intros, outcome lines) never names an object before the choice, never contradicts the engine's outcome, and never reaches the child.

## Using the real models (P4)

1. Copy `.env.example` to `.env.local` (git-ignored; a git-ignored `.env` is read too) and fill in `MUSE_API_KEY` and `MUSE_BASE_URL`. Never put keys in `config/` or in chat.
2. `GROWING_LLM=live uv run python scripts/llm_check.py` – one Director and one Narrator request, shows what came back.
3. `GROWING_LLM=live uv run python -m server.cli new --seed 1` – play; every usable answer is recorded in `cache/llm_cache.jsonl`.
4. `GROWING_LLM=replay ...` replays a recorded game offline and identically (demo without network).
5. `scripts/exp_llm.py --client live --max-calls 200` measures usable / first-try / fallback rates and latency.
Models, cache path and parameters: `config/llm.yaml`.

The models are reasoning models: thinking tokens are spent out of `max_tokens`, so `config/llm.yaml` gives the Director
and the Narrator their own budgets (`agent_params`) and a `reasoning_effort`. Experiment knobs (default = shipped
behaviour): `GROWING_ONTOLOGY` (another ontology file, e.g. `config/experiments/onto_wide.yaml`), `GROWING_DECOYS` /
`GROWING_QUIZ_DECOYS` (`random` | `hard` wrong options for training / the quiz). Results: docs/WORLD.md section 16.
