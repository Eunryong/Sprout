# Growing verification map

The maintained source for verifying what a player of the Growing web game (`python -m server.web`) can do. Read this index, then the feature file that matches the change, then drive it with `../SKILL.md`.

## Baseline preconditions

- Launch with `../scripts/launch.sh 1`: mock LLM, English, free port, fresh save in `saves/verify/<run-id>/save`.
- The child starts at 36 months (3 years) in the default 0-7 years world; the story ends at 84 months.
- `../scripts/doctor.sh "$RUN_DIR"` passes (mode `mock`, lang `en`, our pid owns the port).
- Never touch `saves/web/*`, `saves/children/*`, port 8000, or `.env`.

## Driving conventions

- Start every recipe from a fresh launch unless its preconditions say otherwise. Scenes depend on the seed (`1`), so scene text is stable for a given seed and code, but do not assert on exact prose; assert on structure (option count, ids shown or hidden, counters).
- Use the ids and `data-part` attributes in `../SKILL.md`; do not click by coordinates or by localised label text.
- Wait for `#busy` to be hidden and for `#choose` / `#after` / `#end` to be visible. No fixed sleeps.
- The page and the API both write to the save after every scene; reading `/api/info` `lived` is the second view that proves a scene was learned.

## Proof and skip reporting

- Evidence goes in `$RUN_DIR/evidence/<feature-id>/` and is never deleted by cleanup.
- UI proof: screenshot (page title and `#sceneNo` visible) plus the text of the section that changed. API proof: the request, status code and body. Save-side proof: `lived` before and after.
- Record the feature id and the entry point with every artifact.
- Report what could not be reached with the command tried and the unmet precondition.

## Feature entry contract

Each file: H1, one paragraph of player-visible behaviour, then exactly four H2s in order: `Sub-features`, `How to get to it (user POV)`, `Driving it with <harness>`, `Gotchas`.

## Features

- [Play a scene](./scene-play.md): say / show / leave on a scene, the result card, the child's solo moment, the next scene.
- [No grading](./no-grading.md): the player never sees right/wrong, grade names or odds; `?research=1` does.
- [Life stages and the end](./life-end.md): stage banners, age counter, the end screen with what the child remembers.
- [New child and language](./new-child-language.md): start another child in the same world, switch English / Korean.
- [Talk to the child](./child-qa.md): the question panel and its history.

## Not covered yet

- Taking over a raised child (`/api/children`, `/api/new {"child": ...}`): needs a raised child and may restart the server process in another world.
- Raising a new child in the background (`/api/raise`): runs `server.game.raise_child` for minutes.
- `--new-world` and live Director/Narrator: need the live API key; out of scope for mock verification.
