# No grading

The game never tells the player an option was right or wrong. The page shows an action, what came of it, what the child remembers, and what it seems drawn to next. Grade names, odds and the child's raw expectations appear only in the researcher view (`?research=1`).

## Sub-features

- `grade-hidden` the default page shows no grade words, no odds, no expectation bars, no legend.
- `grade-research` `?research=1` shows the legend, expectation bars and the `research:` lines.
- `grade-result-card` the result card ends with what the child remembers, not with `-> solved`.

## How to get to it (user POV)

- Play any scene on `/` (the player view).
- Open `/?research=1` (the researcher view).

## Driving it with browser

Preconditions:

- Fresh launch, doctor passes. Two scenes are played so that a memory line exists for the second scene's options.

- **Player view.** Open `$URL/`. `#legend` is hidden. No `#opts .bar` element exists. Get the text of `#choose` and, after one play, of `#after`. Neither contains the whole words `solved`, `partial`, `nothing`, `worse`, `right`, `wrong`, `correct`, nor a percentage such as `83%`, nor the text `research:`.
- **Result card.** After a play, the `#result .learned` paragraph exists. It must not end with an arrow and a grade (`-> solved`, `-> worse`, ...). If it does, `remembers` came back empty and the page fell back to `r.learned`: report that as a failure with the scene and the request body.
- **Lean lines.** On later scenes option cards may show "The child seems drawn to this" or "...curious about this" only in research view today (they are rendered under `isResearch`); in the player view they should not show a number.
- **Researcher view.** Open `$URL/?research=1`. `body` has class `research`, `#legend` is visible, option cards have a `.bar` and a `guess ... %` line, results have `research:` lines. Screenshot it.
- **Proof.** Two screenshots (player, research) of the same scene and the saved texts, under `evidence/no-grading/`.

## Gotchas

- The JSON from `/api/scene` and `/api/play` always contains `result` and `expects`. That is expected; judge the rendered page, not the JSON.
- "nothing" and "right" are ordinary English words: search for them as whole words inside the result/option text, and read the hits before reporting. The check is for grade usage, not a bare substring match.
- The Korean page has its own words; a language switch changes which words to look for (`정답`, `맞`, `틀`, `해결`). Verify in `en` unless the change is about Korean text.
- The fallback to `r.learned` in `resultCard` (`index.html`) is the one known place where a grade could leak into the player view.
