# veto — an example cai extension

Refusing tool calls from a `before_tool_call` hook with `cai.HookResult.veto`.

- **`deny_destructive`** — a fixed deny list. The tool never runs; the model
  reads the reason in place of the tool result and carries on.
- **`protect_paths`** — asks the human before a write tool touches `.git`,
  `.env` or `secrets`, through `ctx.ui.confirm`. With no human reachable
  (`ctx.ui.interactive` is False) it vetoes instead of guessing.

## Layout

```
veto/
├── README.md
└── init.py      # @cai.hook("before_tool_call") deny_destructive + protect_paths
```

## Install

```sh
cp -r examples/extensions/veto ~/.config/cai/extensions/
```

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `('before_tool_call', 'deny_destructive')` and
`('before_tool_call', 'protect_paths')` among the hooks.

## How it works

- A hook acts by returning a `cai.HookResult`. `HookResult.veto(reason)` skips
  the tool; the model sees `Error: tool 'x' was aborted by a before_tool_call
  hook: <reason>` as the tool result, so a good reason steers it.
- Returning `None` observes: the call goes ahead. Most hooks return `None` most
  of the time.
- Several hooks on one event are merged: any veto wins, the last reason given
  is the one the model reads.
- `ctx.tool_call` carries the name, the raw argument string and the parsed
  `args` dict; `ctx.messages` is the live conversation if the decision needs
  context.

## Notes

- A hook exception is logged and skipped, which lets the call through. A gate
  that must fail closed catches its own errors and returns a veto.
- See `judge` for the same gate decided by a System One model instead of rules.
