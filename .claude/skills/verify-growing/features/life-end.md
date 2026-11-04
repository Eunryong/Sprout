# Life stages and the end

The child grows from 3 to 7 years. When a new life stage begins the first scene opens with an introduction, the header shows the age and stage, and after the last scene an end screen shows the age, how many scenes were lived, how the parent took part, and what the child remembers.

## Sub-features

- `life-header` the scene counter shows age and number; the stage chip shows the stage name.
- `life-stage-banner` a stage's first scene shows `#banner` with its name and blurb.
- `life-end-screen` after the last scene `#end` shows the summary and no options.
- `life-summary-api` `/api/summary` matches the end screen.
- `life-refuse-after-end` playing after the end is refused.

## How to get to it (user POV)

- Keep playing scenes on `/` until the child is grown (about 48 scenes from 36 months; read `age_months` from `/api/scene` to confirm).
- Reload `/` after the end: it opens on the end screen.

## Driving it with HTTP, then browser

Preconditions:

- Fresh launch, doctor passes.

- **Fast-forward through the real API.** Play every scene with `leave` the way the page would:
  ```bash
  for i in $(seq 1 200); do
    S=$(curl -s $B/api/scene); [ "$(echo "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin)["ended"])')" = True ] && break
    curl -s -o /dev/null -X POST $B/api/play -H 'content-type: application/json' -d '{"part":"leave"}'
  done
  ```
  Save every `/api/scene` `age`, `stage.name` and `stage_first` in a file to show the stage changes happened in order and each introduction appeared once.
- **Summary.** `curl -s $B/api/summary`: `scenes` equals the number of finished scenes, `took_part.left` equals the number of `leave` calls, `moments` is a non-empty list of plain sentences, `stages` lists each stage once.
- **End screen.** Open `$URL/`. `#end` is visible, `#choose` is hidden, `#life` has the age heading, the lived-scenes sentence and the memory list. Screenshot it and save the `#life` text.
- **Refused after the end.** `curl -s -o /dev/null -w '%{http_code}' -X POST $B/api/play -H 'content-type: application/json' -d '{"part":"leave"}'` returns `409`.
- **Stage banner in the page.** Relaunch (fresh) and play in the browser until `#banner` is visible; note `#stageChip`; this takes a few scenes to the next stage, so cap the run at 30 scenes and report if none appeared.

## Gotchas

- A stage introduction shows only on the first scene of a stage and only once per life; reloading must not repeat it.
- The end summary has no grades by design; the memory sentences are the latest of each trouble.
- Reaching the end takes dozens of scenes; use the API loop for the full life and use the browser only for the first scenes, the first banner and the end screen.
- `/api/summary` is only meaningful once `ended` is true.
