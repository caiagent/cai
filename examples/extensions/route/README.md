# route — an example cai extension

Choosing the model per call from a `before_turn` hook, by mutating the agent:
`cai.current_agent().set_model(...)`.

- **`route`** — a small model while the prompt is short, a large one once the
  last call's `prompt_tokens` passes `LARGE_AT` (60k). Runs before every model
  call, first one included.
- **`:route small|large`** — the same switch by hand, from the TUI.

## Layout

```
route/
├── README.md
└── init.py      # @cai.hook("before_turn") route + @cai.command route_cmd
```

## Install

```sh
cp -r examples/extensions/route ~/.config/cai/extensions/
```

Add the two model ids to `~/.config/cai/config.json`:

```json
"route_small_model": "openai/gpt-4o-mini",
"route_large_model": "anthropic/claude-sonnet-4"
```

With either missing the hook stays quiet.

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
print('commands:', sorted(env.commands()))
"
```

Expect `('before_turn', 'route')` and the `route` command.

## How it works

- `before_turn` is an observer: it supports no `HookResult` field. It acts by
  mutating the agent. `cai.current_agent()` is the agent whose run is firing
  the hook, the same accessor a tool uses.
- The loop never copies the model. Every model call, hook context and gated
  tool dispatch resolves it at its own moment, so `set_model` from any hook,
  command or thread is seen by the next read. A switch made in `before_turn`
  is the model that call uses.
- `ctx.usage` is the previous call's usage (`None` on the first call). It is
  the cheapest signal for context size; `cai.models.ModelsRegistry` has the
  context length per model if you want a ratio instead of a count.
- The command drives the served agent over the wire with
  `ctx.client.set_model(...)`, because a command runs on the client side and
  the agent may be another process. Both paths end in `Agent.set_model`.

## Notes

- Other rules fit the same hook: a cheap model for tool-heavy turns and a strong
  one for the final answer, a reasoning model once a `judge`-style question
  says the task is hard, a local model when the prompt contains secrets.
- Switching models mid-conversation changes tokenisation, so the next
  `prompt_tokens` can jump; compare against the same model's numbers.
