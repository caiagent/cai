"""init.py - example extension: end a run from a hook with HookResult.finish.

after_turn fires after every tool-using turn, with the live conversation. two
hooks watch it for the classic failure modes of an agent loop and end the run
cleanly when they see one - no further model call, and the normal final path
still runs (on_final_response hooks, the assistant message on the transcript,
after_run), exactly as if the model had answered:

- stop_when_stuck: the same tool called with the same arguments REPEAT_LIMIT
  times in a row since the last user turn.
- stop_at_budget: TURN_BUDGET tool turns since the last user turn without a
  final answer.

compare with agent.interrupt: that returns the partial text and skips the
final hooks. finish is the graceful one."""
import cai


REPEAT_LIMIT = 3
TURN_BUDGET = 25


@cai.hook("after_turn")
def stop_when_stuck(ctx: cai.HookContext):
    calls = _tool_calls_since_last_user(ctx.messages)
    if len(calls) < REPEAT_LIMIT: return None
    tail = calls[-REPEAT_LIMIT:]
    for call in tail:
        if call != tail[0]: return None
    name, arguments = tail[0]
    return cai.HookResult.finish(f"stopped: {name} was called with the same arguments "
                                 f"{REPEAT_LIMIT} times in a row ({arguments})")


@cai.hook("after_turn")
def stop_at_budget(ctx: cai.HookContext):
    turns = _tool_turns_since_last_user(ctx.messages)
    if turns < TURN_BUDGET: return None
    return cai.HookResult.finish(f"stopped after {turns} tool turns without a final answer")


def _since_last_user(messages):
    """the messages after the most recent user turn, in order."""
    tail = []
    for message in reversed(messages):
        if message.get("role") == "user": break
        tail.append(message)
    tail.reverse()
    return tail


def _tool_calls_since_last_user(messages):
    """(name, arguments) of every tool call the model made since the last user
    turn, in order."""
    calls = []
    for message in _since_last_user(messages):
        if message.get("role") != "assistant": continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            calls.append((function.get("name"), function.get("arguments")))
    return calls


def _tool_turns_since_last_user(messages):
    turns = 0
    for message in _since_last_user(messages):
        if message.get("role") != "assistant": continue
        if not message.get("tool_calls"): continue
        turns += 1
    return turns
