"""init.py - example extension: a tool gate decided by System One (cai.decide).

one before_tool_call hook asks a typed yes/no question - "would this call
destroy or leak the user's data?" - and vetoes above a threshold. no text is
parsed: cai.decide returns a probability. read-only tools are skipped so the
judge is only paid for calls that can change something.

the judge fails CLOSED: if the decision api cannot be reached, or no
`system_one_model` is configured, the call is vetoed with a reason that says
so - an unreachable judge must never wave a call through."""
import cai


THRESHOLD = 0.8

READ_ONLY = ("fs__read_file", "fs__list_files", "fs__search")

QUESTIONS = {}
QUESTIONS["risky"] = {"type": "noul",
                      "instructions": ("Would running this tool call, with these "
                                       "arguments, destroy, overwrite or leak the "
                                       "user's data?")}


@cai.hook("before_tool_call")
def judge(ctx: cai.HookContext):
    name = ctx.tool_call.name
    if name in READ_ONLY: return None

    state = {}
    state["tool"] = name
    state["arguments"] = ctx.tool_call.args
    state["last_user_message"] = _last_user_text(ctx.messages)
    try:
        answers, usage = cai.decide(state, QUESTIONS)
    except (cai.ApiError, ValueError) as e:
        return cai.HookResult.veto(f"the judge could not assess this call: {e}")

    risk = answers["risky"]["noul"]
    if risk < THRESHOLD: return None
    ctx.ui.status(f"judge: {name} vetoed, risk {risk:.2f}")
    return cai.HookResult.veto(f"judged too risky ({risk:.2f} >= {THRESHOLD}) to run "
                               f"unattended; ask the user to do it")


def _last_user_text(messages):
    for message in reversed(messages):
        if message.get("role") != "user": continue
        content = message.get("content")
        if isinstance(content, str): return content
    return ""
