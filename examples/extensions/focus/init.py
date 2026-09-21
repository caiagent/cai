"""init.py - example extension: narrow the agent's tools from a hook.

when the user's latest message says "read-only" (or "read only") the write
tools are parked - deselected on the agent with set_selected_tools - and
re-selected when a later message no longer says so. a before_turn hook, acting
by mutating the agent (cai.current_agent()), the way `route` switches models.

the loop reads the tool list right before every model call, never from a copy
(the same contract as the model), so a selection changed here is what this
very call offers - and dispatch refuses a deselected tool should the model
still name it. `:tools` in the tui shows the live selection."""
import cai


WRITE_TOOLS = ("fs__create_file",
               "fs__edit_file",
               "fs__rename_file",
               "fs__move_file",
               "fs__copy_file",
               "fs__remove_file",
               "fs__create_directory",
               "fs__move_directory")

_parked = []


@cai.hook("before_turn")
def focus(ctx: cai.HookContext):
    agent = cai.current_agent()
    if agent is None: return None
    wants_read_only = _asks_read_only(_last_user_text(ctx.messages))
    selected = list(agent.get_selected_tools())
    if wants_read_only:
        keep = []
        for name in selected:
            if name in WRITE_TOOLS:
                if name not in _parked: _parked.append(name)
                continue
            keep.append(name)
        if keep == selected: return None
        agent.set_selected_tools(keep)
        ctx.ui.status(f"focus: parked {len(selected) - len(keep)} write tools")
        return None
    if not _parked: return None
    restored = selected + _parked
    _parked.clear()
    agent.set_selected_tools(restored)
    ctx.ui.status("focus: write tools restored")
    return None


def _asks_read_only(text):
    lowered = text.lower()
    if "read-only" in lowered: return True
    if "read only" in lowered: return True
    return False


def _last_user_text(messages):
    for message in reversed(messages):
        if message.get("role") != "user": continue
        content = message.get("content")
        if isinstance(content, str): return content
    return ""
