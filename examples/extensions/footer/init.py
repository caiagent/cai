"""init.py - example extension: rewrite the final answer with HookResult.answer.

on_final_response fires once the model has answered with no tool calls, with
the answer in ctx.content. the hook returns HookResult.answer(text) and `text`
replaces it - on the transcript and in what the caller gets back. this one
trims trailing whitespace and appends a one-line footer with the model and the
token count of the run's last call."""
import cai


@cai.hook("on_final_response")
def footer(ctx: cai.HookContext):
    text = (ctx.content or "").rstrip()
    usage = ctx.usage or {}
    tokens = usage.get("total_tokens")
    if tokens is None:
        return cai.HookResult.answer(text)
    return cai.HookResult.answer(f"{text}\n\n-- {ctx.model}, {tokens} tokens")
