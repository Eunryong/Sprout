# Sprout — raise a tiny mind

A 3-to-7 raising game. A randomly initialised ~110k-parameter transformer (the child)
learns for real from lived outcomes. There are no right answers: every choice is an
action, every action has an outcome, and the child learns what happened.

## What it is

You raise one child from age 3 to 7 (48 scenes over 8 days, 6 scenes a day).
Each scene offers a situation with choices; you act through say / show / leave, the
engine decides the outcome, and the child learns from it on the spot.
No grades or expectations are ever shown to the player — only what happened and what
the child remembers. Current direction: [docs/DIRECTION.md](docs/DIRECTION.md).

## Quick Start

```bash
uv sync
uv run pytest
uv run python -m server.web          # open http://127.0.0.1:8000
```

The web server defaults to port 8000 on 127.0.0.1 (`--port`, `--host` to change).
Words and scenes default to `mock` (offline stand-ins, see `mode` in `config/llm.yaml`);
set `GROWING_LLM=live` for the real Director/Narrator. A restart resumes
`saves/web/current` unless `--fresh` is given. Useful variants:

```bash
uv run python -m server.web --small                    # five-problem world
uv run python -m server.web --world saves/worlds/w7/ontology.yaml
uv run python -m server.web --new-world --seed 31      # Director makes a world (a minute or two)
```

## Layout

| Path | What |
|---|---|
| `config/ontology_age07.yaml` | the 0–7 world: problems, verbs, objects, rules table |
| `server/game/simple.py` | the scene loop (say / show / leave → act → outcome → learn) |
| `server/web/` | the web game (`python -m server.web`, FastAPI + static page) |
| `server/agents/` | Director (situations/worlds) and Narrator (prose), with mock fallback |
| `scripts/` | research experiments (multi-seed gates, world checks) |

## Honesty rules

- No grades shown to the player: no solved/partial/nothing/worse labels, no expectation numbers.
- The parent acts once per scene: say, show, or leave. No praise, no corrections.
- Only the engine's rules table says what each action does. Training code never reads it (AST test).
- The child sees tags plus one action (verb + target attributes) — never kinds, values, outcomes, or narration.

## Real models

1. Copy `.env.example` to `.env.local` and fill in `MUSE_API_KEY` and `MUSE_BASE_URL`.
2. `GROWING_LLM=live uv run python scripts/llm_check.py` — one Director + one Narrator request.
3. `GROWING_LLM=replay ...` replays a recorded run offline and identically.

Models, cache path, and budgets: `config/llm.yaml`.

## Research

Background and experiment log: [docs/WORLD.md](docs/WORLD.md). Three entry points:

```bash
uv run python scripts/exp_gate.py --seeds 200-239 --days 10 --grid "pd8|per_day=8"
uv run python scripts/exp_worlds.py --worlds 0-19 --out scripts/out/worlds.json
uv run python scripts/exp_llm.py --client live --max-calls 200
```
