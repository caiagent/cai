# stuck — an example cai extension

Ending a run from a hook with `cai.HookResult.finish`, on `after_turn`.

- **`stop_when_stuck`** — the model called the same tool with the same
  arguments `REPEAT_LIMIT` (3) times in a row: finish with a note saying so.
- **`stop_at_budget`** — `TURN_BUDGET` (25) tool turns since the last user
  message without a final answer: finish.

Both stop the loop before the next model call. The text they pass becomes the
run's answer and takes the normal final path: `on_final_response` hooks run on
it, it lands on the transcript as the assistant message, `after_run` fires.

## Layout

```
stuck/
├── README.md
└── init.py      # @cai.hook("after_turn") stop_when_stuck + stop_at_budget
```

## Install

```sh
cp -r examples/extensions/stuck ~/.config/cai/extensions/
```

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `('after_turn', 'stop_when_stuck')` and `('after_turn', 'stop_at_budget')`.

## How it works

- `after_turn` fires after every tool-using turn with `ctx.messages`, the live
  conversation. The hooks read the tail since the last user turn: assistant
  messages carry `tool_calls`, each with a `function.name` and the raw
  `function.arguments` string, so "same call" is a plain tuple comparison.
- `HookResult.finish(text)` ends the run with `text`. Several hooks on one
  event merge: any finish wins, the last text given is the answer.

## Notes

- This is the graceful stop. `cai.current_agent().interrupt.set()` also ends a
  run, but returns the partial text and skips the final hooks — use it for a
  kill, not a verdict.
- A `noul` question through `cai.decide` ("is the model making progress?") can
  replace the repeat rule; see `judge` for the call shape.
