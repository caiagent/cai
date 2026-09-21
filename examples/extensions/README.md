# Example extensions

Each directory is a self-contained bundle: `cp -r` it under
`~/.config/cai/extensions/` (or `cai extend <dir>`) and `Environment.load()`
picks it up. Every example has a `README.md` with a no-API way to verify it
loaded.

## By hook event

A hook returns `None` to observe, or a `cai.HookResult` to act. Observers act
by mutating: the conversation through `ctx.messages`, the agent through
`cai.current_agent()`.

| event | supports | example | what it shows |
|---|---|---|---|
| `before_turn` | observer | `route` | `current_agent().set_model(...)` picks the model for the call |
| `before_turn` | observer | `focus` | `set_selected_tools(...)` narrows the tools for the call |
| `before_tool_call` | `HookResult.veto(reason)` | `veto` | rules + `ctx.ui.confirm`; the model reads the reason |
| `before_tool_call` | `HookResult.veto(reason)` | `judge` | the same gate decided by `cai.decide` (System One), fail closed |
| `after_turn` | `HookResult.finish(text)` | `stuck` | end the run cleanly on a repeat loop or a turn budget |
| `after_turn` | observer | `nudge` | append a user reminder to the live conversation |
| `after_turn` | observer | `compact` | fold old turns, then `set_messages(...)` so autosave sees it |
| `on_final_response` | `HookResult.answer(text)` | `footer` | rewrite the answer |

`after_tool_call`, `messages_mutated`, `messages_loaded` and `after_run` are
observers too; `compact`'s README shows `after_run` as a second trigger.

## By agent mutation

| call | example |
|---|---|
| `set_model` | `route` (hook) and `:route` (command, via `ctx.client`) |
| `set_selected_tools` | `focus` |
| `set_messages` | `compact` |
| `ctx.messages.append` | `nudge` |
| `interrupt.set()` | noted in `stuck` — the hard stop, vs `finish` |
| `set_selected_skills`, `set_system_prompt_base` | same shape as `focus`; live like the tools (the prompt is resolved before every call) |
| `set_paths` | same shape; republishes grants and restarts MCP servers; not yet shown |

## Other examples

- `ask` — tools that question the human through `cai.current_ui()`.
- `memory` — a `{{memory__notes}}` slot pushes a note store into the prompt.
- `clone` / `summarize` — `:`-commands that checkpoint and branch a session.
