# footer — an example cai extension

Rewriting the final answer from an `on_final_response` hook with
`cai.HookResult.answer`.

- **`footer`** — trims trailing whitespace and appends `-- <model>, <n> tokens`.

## Layout

```
footer/
├── README.md
└── init.py      # @cai.hook("on_final_response") footer
```

## Install

```sh
cp -r examples/extensions/footer ~/.config/cai/extensions/
```

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `('on_final_response', 'footer')`.

## How it works

- `ctx.content` is the model's answer; `HookResult.answer(text)` replaces it.
  The replacement is what lands on the transcript as the assistant message and
  what `run.text` / the CLI print.
- Several hooks on the event merge: the last answer given wins, so order your
  registrations if two rewrite.
- A finish from an `after_turn` hook (see `stuck`) also passes through here, so
  a footer is appended to those too.

## Notes

- The same hook is the place for redaction (strip secrets before they reach
  the terminal or a session file), format enforcement, or a translation pass.
- `on_final_response` does not fire on an interrupted run; the partial text is
  returned as is.
