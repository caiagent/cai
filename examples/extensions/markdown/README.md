# markdown — an example cai extension

Rendering the assistant's markdown in the TUI with a `cai.renderer` — the
display-side extension point. cai itself paints plain text; this bundle turns
headings, lists, quotes, fenced code, tables and inline emphasis into styled
terminal text, live as the answer streams.

- **`markdown`** — a renderer for the `response` kind (the assistant's messages).

## Layout

```
markdown/
├── README.md
└── init.py      # @cai.renderer("response") markdown
```

## Install

```sh
cai extend examples/extensions/markdown
# or: cp -r examples/extensions/markdown ~/.config/cai/extensions/
```

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('response renderers:', [f.__name__ for f in env.renderers('response')])
"
```

Expect `['markdown']`. Then `cai` and ask for anything with a list or a code
block in it.

## How it works

- `@cai.renderer("response")` registers the function on the Environment the TUI
  loads at startup, the way `@cai.hook` and `@cai.command` do. A renderer may
  claim several kinds at once — `@cai.renderer("response", "reasoning")` — out of
  `user`, `response`, `reasoning`, `tool`, `meta`, `error`.
- The transcript calls it with a `cai.RenderContext` holding the **whole**
  block so far (`ctx.text`), not the chunk: on every streamed chunk with
  `ctx.done` False, and once more with `ctx.done` True when the message ends.
  What it returns replaces the block on screen, in place — so the markdown
  takes shape as it streams and is rendered a final time when complete.
- `ctx.style` is the block's base color. Every styled span here closes with a
  reset and re-opens that style, so the surrounding text stays the
  transcript's assistant color. `ctx.width` is the columns the block may use;
  tables are aligned against it, everything else wraps on the screen.
- Renderers of one kind chain in registration order, each seeing the previous
  one's output, so a later bundle (or your `init.py`) can post-process this
  one. Returning `None` leaves the text untouched; a renderer that raises is
  logged and skipped, never breaking the transcript.
- Renderers are a client-side concern and never touch the conversation: the
  stored messages, a saved session and `--tail` all keep the raw markdown.
  Search (`/`) and yank in the TUI see the rendered text.

## Notes

- It runs on every chunk, so keep a renderer linear in the block size; this
  one is a single pass over the lines.
- `:redraw` and a loaded session paint through the same renderer, each block
  final.
- Resizing the terminal rewraps the rendered lines but does not re-render, so
  a table laid out for a wide terminal stays wide until the next `:redraw`.
- To render reasoning too, change the decorator to
  `@cai.renderer("response", "reasoning")`.
