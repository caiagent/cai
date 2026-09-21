"""init.py - example extension: steer the model mid-run by editing the
conversation from an after_turn hook.

ctx.messages is the agent's live conversation - the very list the next model
call reads - so an observer hook can append to it. every EVERY tool turns since
the last user message this one adds a short user reminder to wrap up, and posts
a note on the status line. returning None: it acts by mutation, not by result.

the softer sibling of `stuck`: a nudge asks the model to finish; finish makes
it."""
import cai


EVERY = 8


@cai.hook("after_turn")
def nudge(ctx: cai.HookContext):
    turns = _tool_turns_since_last_user(ctx.messages)
    if turns == 0: return None
    if turns % EVERY != 0: return None
    reminder = (f"[nudge] {turns} tool turns so far. if you have what you need, "
                f"answer now; otherwise say in one line what is still missing.")
    ctx.messages.append({"role": "user", "content": reminder})
    ctx.ui.status(f"nudge after {turns} tool turns")
    return None


def _tool_turns_since_last_user(messages):
    """assistant tool-call turns after the most recent REAL user message - the
    nudges this hook appended are skipped so they do not reset the count."""
    turns = 0
    for message in reversed(messages):
        role = message.get("role")
        if role == "user":
            content = message.get("content")
            if isinstance(content, str) and content.startswith("[nudge]"): continue
            break
        if role != "assistant": continue
        if not message.get("tool_calls"): continue
        turns += 1
    return turns
