# focus — an example cai extension

Narrowing the agent's tool selection from a `before_turn` hook, by mutating
the agent: `cai.current_agent().set_selected_tools(...)`.

- **`focus`** — when the latest user message says "read-only" or "read only",
  the write tools are parked (deselected); when a later message no longer says
  so they are re-selected.

## Layout

```
focus/
├── README.md
└── init.py      # @cai.hook("before_turn") focus
```

## Install

```sh
cp -r examples/extensions/focus ~/.config/cai/extensions/
```

## Verify it loaded (no API call)

```sh
python3 -c "
from cai.environment import Environment
env = Environment().load()
print('hooks:', [(e, f.__name__) for e, f in env.hooks()])
"
```

Expect `('before_turn', 'focus')`.

## How it works

- `agent.get_selected_tools()` is the live selection; `set_selected_tools`
  diffs against it, deselecting what is gone and selecting what is new. A
  deselected tool stays registered, so re-selecting by name works.
- The same pair exists for skills: `get_selected_skills` /
  `set_selected_skills`. A skill brings its tools (selected the same live way)
  and its prompt body: the system prompt is resolved before every call too,
  so the skill's text is in the very next call. `set_system_prompt_base`
  changes the base part the same way.

## Notes

- **The tool list is live.** Like the model (see `route`), the loop resolves
  the tool schemas right before every call and never copies them, so a
  selection changed in `before_turn` is what that call offers. Dispatch is
  live too: a deselected tool is refused even if the model still names it.
- Path policy is another mutation with the same shape:
  `agent.set_paths(allowed=[...], disallowed=[...])` republishes the grants and
  restarts long-lived MCP servers under them.
