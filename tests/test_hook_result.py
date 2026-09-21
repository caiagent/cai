"""cai.HookResult - the one type every hook returns to act. fire() merges the
hooks' results into one; a field the event does not support, or a return that is
not a HookResult, is logged and ignored. the loop reads the merged result:
a veto's reason reaches the model as the tool result, an answer replaces the
final text. fully offline: a fake api streams canned turns."""
import logging

import cai
from cai.hooks import HookContext, HookEvent, HookResult, HooksRegistry
from cai.llm import call_llm


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

def tool_call(name, call_id="c1", arguments="{}"):
    function = {}
    function["name"] = name
    function["arguments"] = arguments
    call = {}
    call["id"] = call_id
    call["type"] = "function"
    call["function"] = function
    return call


class ToolThenTextApi:
    """turn 1 asks for the `poke` tool, turn 2 answers 'final'."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, model, **kwargs):
        self.calls += 1
        n = self.calls
        def gen():
            if n == 1:
                yield (None, None, [tool_call("poke")], {})
            else:
                yield ("final", None, None, {})
        return gen()


class TextApi:
    def chat(self, messages, model, **kwargs):
        def gen():
            yield ("plain", None, None, {})
        return gen()


def drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def ctx_for(event):
    return HookContext(event=event, messages=[], model="m")


def registry(event, *fns):
    reg = HooksRegistry()
    for fn in fns:
        reg.register(event, fn)
    return reg


def run_with_tool(hooks):
    ran = []

    def dispatch(name, args):
        ran.append(name)
        return "poked"

    messages = [{"role": "user", "content": "go"}]
    tools = [{"type": "function", "function": {"name": "poke"}}]
    events, text = drain(call_llm(messages, "m", ToolThenTextApi(),
                                  tools=tools, tools_dispatch=dispatch, hooks=hooks))
    return ran, messages, text


# --------------------------------------------------------------------------
# the type
# --------------------------------------------------------------------------

def test_constructors_set_only_their_fields():
    assert HookResult.veto() == HookResult(vetoed=True, reason="", content=None)
    assert HookResult.veto("no") == HookResult(vetoed=True, reason="no", content=None)
    assert HookResult.answer("x") == HookResult(vetoed=False, reason="", content="x")
    assert HookResult.finish("x") == HookResult(stop=True, content="x")


def test_exported_from_cai():
    assert cai.HookResult is HookResult
    assert "HookResult" in cai.__all__


# --------------------------------------------------------------------------
# fire() merges
# --------------------------------------------------------------------------

def test_fire_returns_an_empty_result_when_hooks_observe():
    def observe(ctx):
        return None
    result = registry("after_turn", observe).fire(HookEvent.AFTER_TURN, ctx_for("after_turn"))
    assert result == HookResult()


def test_fire_ors_vetoes_and_keeps_the_last_reason():
    def first(ctx):
        return HookResult.veto("one")
    def second(ctx):
        return None
    def third(ctx):
        return HookResult.veto("three")
    result = registry("before_tool_call", first, second, third).fire(
        HookEvent.BEFORE_TOOL_CALL, ctx_for("before_tool_call"))
    assert result.vetoed is True
    assert result.reason == "three"


def test_fire_takes_the_last_answer():
    def first(ctx):
        return HookResult.answer("a")
    def second(ctx):
        return HookResult.answer("b")
    result = registry("on_final_response", first, second).fire(
        HookEvent.ON_FINAL_RESPONSE, ctx_for("on_final_response"))
    assert result.content == "b"


def test_raw_false_is_ignored_and_logged(caplog):
    def old_style(ctx):
        return False
    with caplog.at_level(logging.WARNING, logger="cai"):
        result = registry("before_tool_call", old_style).fire(
            HookEvent.BEFORE_TOOL_CALL, ctx_for("before_tool_call"))
    assert result.vetoed is False
    assert "old_style" in caplog.text
    assert "not a HookResult" in caplog.text


def test_field_the_event_does_not_read_is_ignored_and_logged(caplog):
    def wrong(ctx):
        return HookResult.veto("nope")
    with caplog.at_level(logging.WARNING, logger="cai"):
        result = registry("on_final_response", wrong).fire(
            HookEvent.ON_FINAL_RESPONSE, ctx_for("on_final_response"))
    assert result == HookResult()
    assert "HookResult.vetoed" in caplog.text
    assert "on_final_response does not support" in caplog.text


def test_answer_on_before_tool_call_is_ignored_and_logged(caplog):
    def wrong(ctx):
        return HookResult.answer("stub")
    with caplog.at_level(logging.WARNING, logger="cai"):
        result = registry("before_tool_call", wrong).fire(
            HookEvent.BEFORE_TOOL_CALL, ctx_for("before_tool_call"))
    assert result == HookResult()
    assert "HookResult.content" in caplog.text


def test_observer_events_ignore_every_field(caplog):
    def wrong(ctx):
        return HookResult.veto("x")
    with caplog.at_level(logging.WARNING, logger="cai"):
        result = registry("after_run", wrong).fire(HookEvent.AFTER_RUN, ctx_for("after_run"))
    assert result == HookResult()
    assert "after_run does not support" in caplog.text


def test_fire_ors_stops_and_keeps_the_last_finish_text():
    def first(ctx):
        return HookResult.finish("one")
    def second(ctx):
        return None
    def third(ctx):
        return HookResult.finish("three")
    result = registry("after_turn", first, second, third).fire(
        HookEvent.AFTER_TURN, ctx_for("after_turn"))
    assert result.stop is True
    assert result.content == "three"


def test_answer_on_after_turn_is_ignored_and_logged(caplog):
    def wrong(ctx):
        return HookResult.answer("stub")
    with caplog.at_level(logging.WARNING, logger="cai"):
        result = registry("after_turn", wrong).fire(HookEvent.AFTER_TURN, ctx_for("after_turn"))
    assert result == HookResult()
    assert "content without stop" in caplog.text


def test_finish_on_before_tool_call_is_ignored_and_logged(caplog):
    def wrong(ctx):
        return HookResult.finish("bye")
    with caplog.at_level(logging.WARNING, logger="cai"):
        result = registry("before_tool_call", wrong).fire(
            HookEvent.BEFORE_TOOL_CALL, ctx_for("before_tool_call"))
    assert result == HookResult()
    assert "HookResult.stop" in caplog.text


# --------------------------------------------------------------------------
# the loop reads the merged result
# --------------------------------------------------------------------------

def test_veto_skips_the_tool_and_the_model_reads_the_reason():
    def gate(ctx):
        return HookResult.veto("would destroy user data")
    ran, messages, text = run_with_tool(registry("before_tool_call", gate))
    assert ran == []
    assert text == "final"
    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == ("Error: tool 'poke' was aborted by a "
                                   "before_tool_call hook: would destroy user data")


def test_veto_without_reason_keeps_the_fixed_wording():
    def gate(ctx):
        return HookResult.veto()
    ran, messages, text = run_with_tool(registry("before_tool_call", gate))
    assert ran == []
    assert messages[2]["content"] == "Error: tool 'poke' was aborted by a before_tool_call hook"


def test_raw_false_no_longer_vetoes():
    def old_style(ctx):
        return False
    ran, messages, text = run_with_tool(registry("before_tool_call", old_style))
    assert ran == ["poke"]
    assert messages[2]["content"] == "poked"


def test_answer_replaces_the_final_response():
    def sign(ctx):
        return HookResult.answer(ctx.content + " -- cai")
    messages = [{"role": "user", "content": "hi"}]
    events, text = drain(call_llm(messages, "m", TextApi(),
                                  hooks=registry("on_final_response", sign)))
    assert text == "plain -- cai"
    assert messages[-1] == {"role": "assistant", "content": "plain -- cai"}


def test_raw_str_no_longer_replaces_the_final_response():
    def old_style(ctx):
        return "rewritten"
    messages = [{"role": "user", "content": "hi"}]
    events, text = drain(call_llm(messages, "m", TextApi(),
                                  hooks=registry("on_final_response", old_style)))
    assert text == "plain"


# --------------------------------------------------------------------------
# finish from after_turn
# --------------------------------------------------------------------------

def test_finish_ends_the_run_through_the_final_path():
    fired = []

    def done(ctx):
        return HookResult.finish("stopped early")

    def sign(ctx):
        fired.append("on_final_response")
        return HookResult.answer(ctx.content + " -- cai")

    def after_run(ctx):
        fired.append("after_run")

    hooks = HooksRegistry()
    hooks.register("after_turn", done)
    hooks.register("on_final_response", sign)
    hooks.register("after_run", after_run)

    ran = []

    def dispatch(name, args):
        ran.append(name)
        return "poked"

    api = ToolThenTextApi()
    messages = [{"role": "user", "content": "go"}]
    tools = [{"type": "function", "function": {"name": "poke"}}]
    events, text = drain(call_llm(messages, "m", api,
                                  tools=tools, tools_dispatch=dispatch, hooks=hooks))
    assert ran == ["poke"]                      # the tool turn ran
    assert api.calls == 1                       # no second model call
    assert text == "stopped early -- cai"
    assert messages[-1] == {"role": "assistant", "content": "stopped early -- cai"}
    assert fired == ["on_final_response", "after_run"]


def test_finish_with_none_content_ends_with_an_empty_answer():
    def done(ctx):
        return HookResult(stop=True)
    ran, messages, text = run_with_tool(registry("after_turn", done))
    assert text == ""
    assert messages[-1] == {"role": "assistant", "content": ""}


# --------------------------------------------------------------------------
# cai.current_agent() inside a hook
# --------------------------------------------------------------------------

class FakeAgent:
    name = "the-agent"


def test_fire_publishes_the_agent_from_ctx_data():
    seen = []

    def watch(ctx):
        seen.append(cai.current_agent())

    agent = FakeAgent()
    ctx = HookContext(event="after_turn", messages=[], model="m", data={"agent": agent})
    registry("after_turn", watch).fire(HookEvent.AFTER_TURN, ctx)
    assert seen == [agent]
    assert cai.current_agent() is None


def test_fire_without_an_agent_leaves_current_agent_none():
    seen = []

    def watch(ctx):
        seen.append(cai.current_agent())

    registry("after_turn", watch).fire(HookEvent.AFTER_TURN, ctx_for("after_turn"))
    assert seen == [None]


def test_current_agent_is_reset_after_a_hook_raises():
    def boom(ctx):
        raise RuntimeError("x")

    ctx = HookContext(event="after_turn", messages=[], model="m", data={"agent": FakeAgent()})
    registry("after_turn", boom).fire(HookEvent.AFTER_TURN, ctx)
    assert cai.current_agent() is None
