# Talk to the child

A panel lets the player ask the child a question and read its answer, built from what the child has lived so far. History is saved with the game.

## Sub-features

- `qa-open` the panel opens and closes.
- `qa-ask` a question gets an answer and both appear in the history.
- `qa-history` history survives a reload.

## How to get to it (user POV)

- Press `#openQaBtn`, or `#summaryQaBtn` on the end screen.

## Driving it with browser + HTTP

Preconditions:

- Fresh launch, doctor passes, at least three scenes played (the child has something to remember).

- **Open.** Click `#openQaBtn`. `#qaCard` is visible with the suggestion chips in `#qaChips`.
- **Ask.** Click the first `.qa-chip` (or fill `#qaInput` and click `#qaSendBtn`). `#qaMessages` gains the question and a non-empty answer. `#err` stays hidden.
- **API twin.** `curl -s -X POST $B/api/ask -H 'content-type: application/json' -d '{"question":"What do you remember?"}'` returns 200; `curl -s $B/api/ask` returns a `history` containing it.
- **Reload.** Reload the page, open the panel: the earlier exchange is still in `#qaMessages`.
- **Close.** Click `#qaCloseBtn`; the game scene is visible again.
- **Proof.** Screenshot of the open panel with the exchange, and the `history` JSON, under `evidence/child-qa/`.

## Gotchas

- It was not checked whether `/api/ask` produces a real answer under `GROWING_LLM=mock` or needs a live client. If it fails under mock, record the status code and body, and report the feature as unverified rather than switching to live.
- The panel stays open across scenes (`sessionStorage` key `qaOpen`); close it before driving scene recipes.
- Answers come from the child's own history; asking before any scene was lived gives a thin answer, so play first.
