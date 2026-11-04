# Play a scene

The player reads a situation with several options and takes part once: `say` an option (tell the child to do it), `show` it (do it while the child watches), or `leave` it to the child. The result card says what happened and what the child will remember, the child learns from it immediately, and the next scene appears.

## Sub-features

- `scene-load` shows narration, an age/scene counter and at least two options.
- `scene-show` the parent shows an option; a result card follows.
- `scene-say` the parent tells the child an option (age 3+); the card says whether the child did as told.
- `scene-leave` the child chooses on its own; the card shows what it did.
- `scene-step2` after the parent's part, the child's solo moment (two-step mode): thought cards, then a result.
- `scene-next` `#next` advances; the scene number goes up by one per finished scene.
- `scene-bad-input` an impossible option is refused without learning anything.

## How to get to it (user POV)

- Open `/`. The first scene appears by itself.
- On an option card press its button (`[data-part="say"]` at 36+ months, `[data-part="show"]` otherwise).
- Press the "let the child try" control: `#leave` or the `[data-part="leave"]` card button.
- Press `#next` to move on.

## Driving it with browser + HTTP

Preconditions:

- Fresh launch, doctor passes, `/api/info` `lived` is `0`.

- **Load.** Open `$URL/`. Wait until `#choose` is visible. `#sceneNo` is non-empty, `#story` has text, `#opts li.opt` count is at least 2, `#err` is hidden. Snapshot `#sceneNo`, `#story`, option titles.
- **Show an option.** Click `#opts li.opt:nth-child(1) button[data-part]`. `#after` becomes visible; `#result .who` and the narration paragraph are non-empty; `#result` has a `.learned` paragraph. Save the `#result` text.
- **Second view.** `curl -s $B/api/info`. Record `lived`. Record `/api/scene` `number` before the click and after the whole scene is finished.
- **Step 2 or next.** Click `#next`. If `#stepChip` is visible the child's solo moment is shown (cards in `li.thought-opt`, one `li.solo-opt` button); click it, read the second `#result`, then `#next`. When the scene finishes, `#sceneNo` shows the next scene number and `lived` has gone up by exactly one scene.
- **Leave.** On the next scene click `#leave`. `#result .who` reads as the child acting on its own; `.child-said` may be present.
- **Say.** If `can_act` and `age_months >= 36`, click a `[data-part="say"]` button. The result `.who` says the child followed or did something else.
- **API twin.** `curl -s -X POST $B/api/play -H 'content-type: application/json' -d '{"part":"say","option":0}'` returns 200 with `narration`, `result`, `learned` (ends with the result), `followed` (true/false for `say`, null otherwise).
- **Bad input.** `curl -s -o /dev/null -w '%{http_code}' -X POST $B/api/play -H 'content-type: application/json' -d '{"part":"say","option":99}'` returns `400`; `{"part":"judge"}` returns `422`; `/api/info` `lived` does not change.
- **Proof.** Screenshot after the show and after the leave, plus the saved text of `#story` (before) and `#result` (after), under `evidence/scene-play/`.

## Gotchas

- In two-step mode (the page default) one scene is two plays; assert `lived` and `number` after the whole scene, not after step 1.
- The same `/api/scene` is returned until the parent takes part; reloading the page must not skip a scene.
- A second click while `#busy` is visible is ignored by the page. Do not retry it as a failure.
- Option button labels are the same word for say and show and are localised; use `data-part`.
- Mock narration is filler. Do not judge prose quality here.
- Clicking and POSTing `/api/play` on the same scene both count; do one or the other per scene when counting `lived`.
