# judge — an example cai extension

A tool gate decided by a System One model through `cai.decide`, from a
`before_tool_call` hook. Where `veto` uses fixed rules, `judge` asks one typed
question per call:

    Would running this tool call, with these arguments, destroy, overwrite or
    leak the user's data?

and vetoes when the answer's probability crosses `THRESHOLD` (0.8). Nothing is
parsed: a `noul` answer is a number.

## Layout

```
judge/
├── README.md
└── init.py      # @cai.hook("before_tool_call") judge
```

## Install

```sh
cp -r examples/extensions/judge ~/.config/cai/extensions/
```

Set `system_one_model` in `~/.config/cai/config.json` (e.g. `"jev-latest"`, or
`"~typesafe/jev-latest"` on OpenRouter). Without it every non-read-only call is
vetoed with a reason that says so — the judge fails closed.

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `('before_tool_call', 'judge')`.

## How it works

- The state handed to the model is a small JSON object: the tool name, its
  parsed arguments, and the last user message, so "delete the build dir" and
  "clean up" read differently.
- `cai.decide(state, questions)` returns `(answers, usage)` in the wire shapes;
  `answers["risky"]["noul"]` is the probability.
- `HookResult.veto(reason)` skips the call and the model reads the reason, so
  it can ask the user instead of retrying.
- Read-only tools are skipped up front: a judge call costs money and time, so
  it is only paid for calls that can change something.

## Notes

- **Fail closed.** A hook that raises is logged and skipped, which would let
  the call through. `judge` catches `ApiError` (the endpoint) and `ValueError`
  (no model configured) itself and vetoes with the error in the reason.
- The threshold is a module constant; tune it to the model. Calibration is the
  point of a decision model: 0.8 means 0.8.
- Pair it with `veto` for hard rules that should never reach a model at all.
