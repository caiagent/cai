# nudge — an example cai extension

Steering the model mid-run by editing the live conversation from an
`after_turn` hook. No `HookResult`: the hook acts by mutation and returns
`None`.

- **`nudge`** — every `EVERY` (8) tool turns since the last real user message,
  appends a short user reminder to wrap up and posts a status-line note.

## Layout

```
nudge/
├── README.md
└── init.py      # @cai.hook("after_turn") nudge
```

## Install

```sh
cp -r examples/extensions/nudge ~/.config/cai/extensions/
```

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `('after_turn', 'nudge')`.

## How it works

- `ctx.messages` is the agent's own list, not a copy. Appending a user message
  here is exactly what a steer does: the next model call reads it.
- Nudges are tagged `[nudge]` so the counter skips them and keeps counting from
  the human's last message.
- `ctx.ui.status(...)` writes the TUI status line; headless it is a no-op.

## Notes

- `compact` mutates the same list the other way, replacing a span with a
  summary, and calls `cai.current_agent().set_messages(...)` afterwards so
  listeners (autosave) see the edit. An append needs no such call: the list
  is the conversation.
- To stop instead of nudge, return `cai.HookResult.finish(text)`; see `stuck`.
