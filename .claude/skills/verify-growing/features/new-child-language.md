# New child and language

The player can start another child in the same world (the current one stays saved), and can switch the words the page uses between English and Korean without changing the child.

## Sub-features

- `new-child-restart` `New child` and a temperament start a child at scene 1; the earlier child is kept aside.
- `new-child-cancel` cancelling the chooser changes nothing.
- `lang-switch` the language button changes the page words and `/api/info` `lang`.
- `lang-keeps-child` the switch does not change `lived` or the scene.

## How to get to it (user POV)

- Press `#newGame` (`New child`), pick a temperament in `#adopt`, or `Cancel`.
- Press `#langBtn`.

## Driving it with browser + HTTP

Preconditions:

- Fresh launch, doctor passes, two scenes played (`lived` is 2).

- **Language.** Click `#langBtn`. `GET $B/api/info` returns `"lang":"ko"`; `#langBtn` now reads the other language; `#story` or the next scene's text is Korean. `lived` is still 2. Click it again to return to English.
- **Cancel.** Click `#newGame`, then the cancel button in `#adopt`. `#adopt` hides; `lived` is still 2.
- **New child.** Click `#newGame`, pick the first temperament button in `#adopt`. Wait for `#choose`. `#sceneNo` shows scene 1, `/api/info` `lived` is `0`, and `$RUN_DIR/save-old-*` exists (the earlier child), while `$RUN_DIR/save` is the new one.
- **Proof.** Screenshot before and after, `/api/info` bodies, and `ls "$RUN_DIR"` showing `save-old-*`, under `evidence/new-child-language/`.

## Gotchas

- This run's `children/` dir is empty on purpose, so the temperament route is the only one tested; adopting a raised child may restart the whole process in another world and is not covered here.
- The page remembers its language in `localStorage` and in the server; reset both when a recipe needs a known language (relaunch + clear storage).
- The old save is renamed next to the new one; cleanup removes `save-old-*` but never the evidence.
