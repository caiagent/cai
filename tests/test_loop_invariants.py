"""call_llm's invariants under every hook: whatever a hook returns or does, the
loop keeps its shape.

the shape: messages stay a valid transcript (an assistant turn carrying N tool
calls is followed by exactly N tool messages with matching ids, in order; the
run ends on an assistant message without tool calls; no system message ever
lands in `messages`); hooks fire in a fixed order with the right context; the
final text is what the caller gets; the live getters are read once per call.

the matrix: every event x every result (None, each HookResult constructor, a
raw value, an exception). only the supported (event, field) pairs act -
before_tool_call + veto, on_final_response + answer (a finish there is an
answer with a warned stop), after_turn + finish - and every other pair must
leave the run exactly as if the hook had returned nothing. fully offline."""
import copy
import logging
import threading

import pytest

from cai.events import EventType
from cai.hooks import HookEvent, HookResult, HooksRegistry
from cai.llm import MaxStepsReached, call_llm


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

def wire_call(name, call_id, arguments="{}"):
    function = {}
    function["name"] = name
    function["arguments"] = arguments
    call = {}
    call["id"] = call_id
    call["type"] = "function"
    call["function"] = function
    return call


class ScriptedApi:
    """plays the model from a script: a list of (name, arguments) is a tool
    turn, a str is the text answer. records the messages each call saw."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, model, tools=None, stream=False, **kwargs):
        record = {}
        record["model"] = model
        record["tools"] = tools
        record["messages"] = copy.deepcopy(messages)
        self.calls.append(record)
        n = len(self.calls)
        step = self.script.pop(0)
        usage = {"prompt_tokens": 10 * n, "total_tokens": 10 * n + 1}
        if isinstance(step, str):
            result = (step, None, None, usage)
        else:
            calls = []
            for i, (name, arguments) in enumerate(step):
                calls.append(wire_call(name, f"c{n}_{i}", arguments))
            result = (None, None, calls, usage)
        if not stream:
            return result
        def gen():
            yield result
        return gen()


def drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


TOOLS = [{"type": "function", "function": {"name": "poke"}},
         {"type": "function", "function": {"name": "prod"}}]


def make_dispatch(ran):
    def dispatch(name, args):
        ran.append((name, dict(args)))
        return f"{name}:ok"
    return dispatch


def registry(*pairs):
    reg = HooksRegistry()
    for event, fn in pairs:
        reg.register(event, fn)
    return reg


def run_loop(script, hooks=None, **kwargs):
    """drive call_llm over a fresh conversation with the scripted model.
    returns (api, messages, events, text, ran)."""
    api = ScriptedApi(script)
    ran = []
    messages = [{"role": "user", "content": "go"}]
    events, text = drain(call_llm(messages, "m", api, tools=TOOLS,
                                  tools_dispatch=make_dispatch(ran),
                                  hooks=hooks, system_prompt="SYS", **kwargs))
    return api, messages, events, text, ran


TWO_TURNS = [[("poke", '{"a": 1}'), ("prod", "{}")], [("poke", "{}")], "final"]


def assert_transcript(messages, finished=True):
    """the pairing invariant: every assistant tool_calls turn is followed by
    exactly its tool messages, ids in order; no system role; a finished run
    ends on a plain assistant message."""
    i = 0
    while i < len(messages):
        message = messages[i]
        assert message["role"] != "system"
        calls = message.get("tool_calls")
        if message["role"] == "assistant" and calls:
            for call in calls:
                i += 1
                assert i < len(messages), "tool call without its tool message"
                assert messages[i]["role"] == "tool"
                assert messages[i]["tool_call_id"] == call["id"]
        elif message["role"] == "tool":
            raise AssertionError(f"orphan tool message at {i}")
        i += 1
    if finished:
        assert messages[-1]["role"] == "assistant"
        assert not messages[-1].get("tool_calls")


def outcome(script, hooks):
    """everything observable about a run, for equality with a baseline."""
    api, messages, events, text, ran = run_loop(script, hooks)
    assert_transcript(messages)
    seen = []
    for record in api.calls:
        seen.append(record["messages"])
    kinds = []
    for event in events:
        kinds.append(event.type)
    return {"messages": messages, "text": text, "ran": ran, "calls": seen, "events": kinds}


EVENTS = list(HookEvent)

# the events a run fires; messages_loaded is the Agent's (a session load), so
# a hook on it never runs inside call_llm and leaves no trace in the log.
LOOP_EVENTS = []
for _event in EVENTS:
    if _event == HookEvent.MESSAGES_LOADED: continue
    LOOP_EVENTS.append(_event)

RESULTS = {}
RESULTS["none"] = None
RESULTS["veto"] = HookResult.veto("because")
RESULTS["answer"] = HookResult.answer("rewritten")
RESULTS["finish"] = HookResult.finish("finished")
RESULTS["raw_false"] = False
RESULTS["raw_str"] = "rewritten"
RESULTS["raw_dict"] = {"vetoed": True}

ACTING = {}
ACTING[(HookEvent.BEFORE_TOOL_CALL, "veto")] = True
ACTING[(HookEvent.ON_FINAL_RESPONSE, "answer")] = True
ACTING[(HookEvent.AFTER_TURN, "finish")] = True
# finish carries content, and on_final_response supports content: a finish
# there acts as an answer (its stop is warned and ignored). pinned below.
ACTING[(HookEvent.ON_FINAL_RESPONSE, "finish")] = True


def returning(value):
    def hook(ctx):
        return value
    return hook


def raising(ctx):
    raise RuntimeError("hook boom")


# --------------------------------------------------------------------------
# baseline: the order of things
# --------------------------------------------------------------------------

def test_hook_order_over_a_tool_turn_and_an_answer():
    order = []
    hooks = HooksRegistry()

    def record(ctx):
        order.append(str(ctx.event))
    for event in EVENTS:
        hooks.register(event, record)

    run_loop([[("poke", "{}"), ("prod", "{}")], "final"], hooks)
    assert order == ["before_turn",
                     "before_tool_call", "messages_mutated", "after_tool_call",
                     "before_tool_call", "messages_mutated", "after_tool_call",
                     "after_turn",
                     "before_turn",
                     "on_final_response", "after_run"]


def test_hook_order_over_a_plain_answer():
    order = []
    hooks = HooksRegistry()

    def record(ctx):
        order.append(str(ctx.event))
    for event in EVENTS:
        hooks.register(event, record)

    run_loop(["final"], hooks)
    assert order == ["before_turn", "on_final_response", "after_run"]


def test_event_stream_shape():
    api, messages, events, text, ran = run_loop(TWO_TURNS)
    kinds = []
    for event in events:
        kinds.append(event.type)
    assert kinds == [EventType.USAGE,
                     EventType.TOOL_CALL, EventType.TOOL_RESULT,
                     EventType.TOOL_CALL, EventType.TOOL_RESULT,
                     EventType.USAGE,
                     EventType.TOOL_CALL, EventType.TOOL_RESULT,
                     EventType.CONTENT, EventType.USAGE]
    assert text == "final"
    assert_transcript(messages)


def test_baseline_transcript_and_dispatch():
    api, messages, events, text, ran = run_loop(TWO_TURNS)
    assert ran == [("poke", {"a": 1}), ("prod", {}), ("poke", {})]
    roles = []
    for message in messages:
        roles.append(message["role"])
    assert roles == ["user", "assistant", "tool", "tool", "assistant", "tool", "assistant"]
    assert messages[-1] == {"role": "assistant", "content": "final"}
    assert api.calls[0]["messages"][0] == {"role": "system", "content": "SYS"}


# --------------------------------------------------------------------------
# the matrix: every event x every result
# --------------------------------------------------------------------------

@pytest.mark.parametrize("event", EVENTS)
@pytest.mark.parametrize("name", list(RESULTS))
def test_non_acting_pairs_leave_the_run_untouched(event, name, caplog):
    if ACTING.get((event, name)): pytest.skip("an acting pair, covered below")
    baseline = outcome(list(TWO_TURNS), None)
    with caplog.at_level(logging.WARNING, logger="cai"):
        got = outcome(list(TWO_TURNS), registry((event, returning(RESULTS[name]))))
    assert got == baseline
    if name == "none": return
    if event not in LOOP_EVENTS: return
    # a mistake is loud in the log, never silent
    assert "ignored" in caplog.text


@pytest.mark.parametrize("event", EVENTS)
def test_a_raising_hook_never_breaks_the_run(event, caplog):
    baseline = outcome(list(TWO_TURNS), None)
    with caplog.at_level(logging.ERROR, logger="cai"):
        got = outcome(list(TWO_TURNS), registry((event, raising)))
    assert got == baseline
    if event in LOOP_EVENTS:
        assert "raised" in caplog.text


def test_veto_acts_only_on_the_tool_it_answers():
    def veto_prod(ctx):
        if ctx.tool_call.name == "prod": return HookResult.veto("no prod")
        return None

    api, messages, events, text, ran = run_loop(TWO_TURNS, registry(("before_tool_call", veto_prod)))
    assert_transcript(messages)
    assert ran == [("poke", {"a": 1}), ("poke", {})]
    assert messages[3]["content"] == "Error: tool 'prod' was aborted by a before_tool_call hook: no prod"
    assert messages[2]["content"] == "poke:ok"
    assert text == "final"
    results = []
    for event in events:
        if event.type != EventType.TOOL_RESULT: continue
        results.append(event.is_error)
    assert results == [False, True, False]


def test_veto_still_fires_the_after_hooks_with_the_error_as_content():
    seen = []

    def after(ctx):
        seen.append((str(ctx.event), ctx.tool_call.name, ctx.content))

    def mutated(ctx):
        seen.append((str(ctx.event), ctx.data["name"], ctx.data["id"]))

    hooks = registry(("before_tool_call", returning(HookResult.veto("nope"))),
                     ("after_tool_call", after),
                     ("messages_mutated", mutated))
    run_loop([[("poke", "{}")], "final"], hooks)
    assert seen == [("messages_mutated", "poke", "c1_0"),
                    ("after_tool_call", "poke", "Error: tool 'poke' was aborted by a before_tool_call hook: nope")]


def test_answer_rewrites_the_text_and_the_transcript_only():
    api, messages, events, text, ran = run_loop(TWO_TURNS, registry(("on_final_response", returning(HookResult.answer("rewritten")))))
    assert text == "rewritten"
    assert messages[-1] == {"role": "assistant", "content": "rewritten"}
    assert ran == [("poke", {"a": 1}), ("prod", {}), ("poke", {})]
    assert len(api.calls) == 3
    # the model's own text still streamed as the CONTENT event
    contents = []
    for event in events:
        if event.type == EventType.CONTENT: contents.append(event.text)
    assert contents == ["final"]


def test_finish_ends_the_run_with_no_further_call_and_the_final_hooks():
    order = []

    def record(ctx):
        order.append(str(ctx.event))
        if ctx.event == HookEvent.ON_FINAL_RESPONSE:
            order.append(f"content={ctx.content}")

    hooks = registry(("after_turn", returning(HookResult.finish("finished"))),
                     ("on_final_response", record),
                     ("after_run", record),
                     ("before_turn", record))
    api, messages, events, text, ran = run_loop(TWO_TURNS, hooks)
    assert len(api.calls) == 1
    assert text == "finished"
    assert_transcript(messages)
    assert messages[-1] == {"role": "assistant", "content": "finished"}
    assert order == ["before_turn", "on_final_response", "content=finished", "after_run"]


def test_finish_on_final_response_acts_as_an_answer_and_warns_about_stop(caplog):
    with caplog.at_level(logging.WARNING, logger="cai"):
        api, messages, events, text, ran = run_loop(TWO_TURNS, registry(("on_final_response", returning(HookResult.finish("finished")))))
    assert text == "finished"
    assert len(api.calls) == 3                   # nothing stopped early
    assert "HookResult.stop" in caplog.text
    assert "on_final_response does not support" in caplog.text


def test_finish_text_still_goes_through_answer():
    hooks = registry(("after_turn", returning(HookResult.finish("finished"))),
                     ("on_final_response", returning(HookResult.answer("signed"))))
    api, messages, events, text, ran = run_loop(TWO_TURNS, hooks)
    assert text == "signed"
    assert messages[-1]["content"] == "signed"


def test_an_acting_result_with_stray_fields_acts_and_warns(caplog):
    both = HookResult(vetoed=True, reason="r", content="stub", stop=True)
    with caplog.at_level(logging.WARNING, logger="cai"):
        api, messages, events, text, ran = run_loop([[("poke", "{}")], "final"],
                                                    registry(("before_tool_call", returning(both))))
    assert ran == []
    assert messages[2]["content"].endswith(": r")
    assert text == "final"
    assert "HookResult.content" in caplog.text
    assert "HookResult.stop" in caplog.text


# --------------------------------------------------------------------------
# several hooks on one event
# --------------------------------------------------------------------------

def test_hooks_fire_in_registration_order_and_merge():
    order = []

    def first(ctx):
        order.append("first")
        return HookResult.answer("one")

    def second(ctx):
        order.append("second")
        return HookResult.answer("two")

    def third(ctx):
        order.append("third")
        return None

    api, messages, events, text, ran = run_loop(["final"], registry(("on_final_response", first),
                                                                    ("on_final_response", second),
                                                                    ("on_final_response", third)))
    assert order == ["first", "second", "third"]
    assert text == "two"


def test_any_veto_wins_over_an_observer_and_a_raiser():
    hooks = registry(("before_tool_call", returning(None)),
                     ("before_tool_call", raising),
                     ("before_tool_call", returning(HookResult.veto("v"))),
                     ("before_tool_call", returning(None)))
    api, messages, events, text, ran = run_loop([[("poke", "{}")], "final"], hooks)
    assert ran == []
    assert messages[2]["content"].endswith(": v")


# --------------------------------------------------------------------------
# hooks that mutate the conversation
# --------------------------------------------------------------------------

def test_appending_in_after_turn_lands_after_the_tool_messages():
    def nudge(ctx):
        ctx.messages.append({"role": "user", "content": "[nudge]"})

    api, messages, events, text, ran = run_loop(TWO_TURNS, registry(("after_turn", nudge)))
    assert_transcript(messages)
    roles = []
    for message in messages:
        roles.append(message["role"])
    assert roles == ["user", "assistant", "tool", "tool", "user",
                     "assistant", "tool", "user", "assistant"]
    # the very next call saw the nudge, after the tool messages
    second_call = api.calls[1]["messages"]
    assert second_call[-1] == {"role": "user", "content": "[nudge]"}
    assert second_call[-2]["role"] == "tool"


def test_appending_in_before_turn_is_seen_by_that_call():
    def prime(ctx):
        if ctx.usage is not None: return None
        ctx.messages.append({"role": "user", "content": "[primed]"})
        return None

    api, messages, events, text, ran = run_loop(["final"], registry(("before_turn", prime)))
    assert api.calls[0]["messages"][-1] == {"role": "user", "content": "[primed]"}
    assert messages == [{"role": "user", "content": "go"},
                        {"role": "user", "content": "[primed]"},
                        {"role": "assistant", "content": "final"}]


def test_replacing_the_conversation_in_place_keeps_the_loop_on_the_same_list():
    """a compaction-style hook: fold everything before the last tool pair into
    one summary. the loop keeps appending to the same list object, so the
    next call sees the fold and the caller's `messages` is the folded one."""
    def fold(ctx):
        if len(ctx.messages) < 5: return None
        summary = {"role": "assistant", "content": "[summary]"}
        keep = ctx.messages[-2:]      # the last assistant(tool_calls) + its tool message
        ctx.messages[:] = [ctx.messages[0], summary] + keep
        return None

    api, messages, events, text, ran = run_loop(TWO_TURNS, registry(("after_turn", fold)))
    assert_transcript(messages)
    assert messages[1] == {"role": "assistant", "content": "[summary]"}
    # the third call saw the fold (index 0 is the system message)
    assert api.calls[2]["messages"][2:4] == [{"role": "assistant", "content": "[summary]"},
                                             messages[2]]
    assert text == "final"


def test_a_mutated_conversation_never_leaks_the_system_message():
    def look(ctx):
        for message in ctx.messages:
            assert message["role"] != "system"

    hooks = HooksRegistry()
    for event in EVENTS:
        hooks.register(event, look)
    api, messages, events, text, ran = run_loop(TWO_TURNS, hooks)
    assert api.calls[0]["messages"][0]["role"] == "system"
    assert_transcript(messages)


# --------------------------------------------------------------------------
# what hooks see
# --------------------------------------------------------------------------

def test_usage_and_content_reach_the_right_hooks():
    seen = {}

    def before_turn(ctx):
        seen.setdefault("before_turn", []).append(ctx.usage)

    def after_turn(ctx):
        seen.setdefault("after_turn", []).append(ctx.usage)

    def final(ctx):
        seen["final"] = (ctx.usage, ctx.content)

    def after_run(ctx):
        seen["after_run"] = (ctx.usage, ctx.content)

    hooks = registry(("before_turn", before_turn), ("after_turn", after_turn),
                     ("on_final_response", final), ("after_run", after_run))
    run_loop([[("poke", "{}")], "final"], hooks)
    assert seen["before_turn"] == [None, {"prompt_tokens": 10, "total_tokens": 11}]
    assert seen["after_turn"] == [{"prompt_tokens": 10, "total_tokens": 11}]
    assert seen["final"] == ({"prompt_tokens": 20, "total_tokens": 21}, "final")
    assert seen["after_run"] == ({"prompt_tokens": 20, "total_tokens": 21}, "final")


def test_before_tool_call_sees_raw_and_parsed_arguments_even_when_invalid():
    seen = []

    def look(ctx):
        seen.append((ctx.tool_call.name, ctx.tool_call.arguments, ctx.tool_call.args))

    api, messages, events, text, ran = run_loop([[("poke", "{not json"), ("prod", '{"x": 2}')], "final"],
                                                registry(("before_tool_call", look)))
    assert seen == [("poke", "{not json", {}), ("prod", '{"x": 2}', {"x": 2})]
    assert ran == [("prod", {"x": 2})]                 # the broken call never dispatched
    assert messages[2]["content"].startswith("Error: arguments for tool 'poke'")
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == "{}"   # echoed clean
    assert_transcript(messages)


def test_hook_contexts_carry_the_model_of_the_moment():
    box = {"model": "m1"}
    seen = []

    def current():
        return box["model"]

    def look(ctx):
        seen.append((str(ctx.event), ctx.model))
        if ctx.event == HookEvent.AFTER_TOOL_CALL: box["model"] = "m2"

    hooks = HooksRegistry()
    for event in EVENTS:
        hooks.register(event, look)
    api = ScriptedApi([[("poke", "{}")], "final"])
    messages = [{"role": "user", "content": "go"}]
    drain(call_llm(messages, current, api, tools=TOOLS, tools_dispatch=make_dispatch([]), hooks=hooks))
    assert seen == [("before_turn", "m1"), ("before_tool_call", "m1"), ("messages_mutated", "m1"),
                    ("after_tool_call", "m1"), ("after_turn", "m2"), ("before_turn", "m2"),
                    ("on_final_response", "m2"), ("after_run", "m2")]
    assert api.calls[0]["model"] == "m1"
    assert api.calls[1]["model"] == "m2"


# --------------------------------------------------------------------------
# the live getters are read per call, exactly
# --------------------------------------------------------------------------

def test_system_prompt_and_tools_getters_are_read_once_per_model_call():
    prompt_reads = []
    tools_reads = []

    def prompt():
        prompt_reads.append(1)
        return "SYS"

    def tools():
        tools_reads.append(1)
        return TOOLS

    hooks = HooksRegistry()
    for event in EVENTS:
        hooks.register(event, returning(None))
    api = ScriptedApi(TWO_TURNS)
    drain(call_llm([{"role": "user", "content": "go"}], "m", api, tools=tools,
                   tools_dispatch=make_dispatch([]), hooks=hooks, system_prompt=prompt))
    assert len(api.calls) == 3
    assert len(prompt_reads) == 3
    assert len(tools_reads) == 3


# --------------------------------------------------------------------------
# interrupt, steer, max_steps under hooks
# --------------------------------------------------------------------------

def test_interrupt_from_a_hook_returns_partial_and_skips_the_final_hooks():
    interrupt = threading.Event()
    order = []

    def record(ctx):
        order.append(str(ctx.event))

    def stop(ctx):
        interrupt.set()

    hooks = registry(("after_turn", stop), ("on_final_response", record), ("after_run", record),
                     ("before_turn", record))
    api = ScriptedApi(TWO_TURNS)
    messages = [{"role": "user", "content": "go"}]
    events, text = drain(call_llm(messages, "m", api, tools=TOOLS, tools_dispatch=make_dispatch([]),
                                  hooks=hooks, interrupt=interrupt))
    assert len(api.calls) == 1
    assert text == ""
    assert order == ["before_turn"]
    assert_transcript(messages, finished=False)
    assert messages[-1]["role"] == "tool"


def test_steer_pending_at_the_answer_reenters_the_loop_with_hooks():
    """the queue is drained at the top of every turn and once more before an
    answer is accepted; a steer that lands during the answer's call is seen
    by that last drain and re-enters the loop."""
    order = []
    drains = []

    def steer():
        drains.append(1)
        if len(drains) != 2: return []      # arrives during the first call
        return ["one more thing"]

    def record(ctx):
        order.append(str(ctx.event))

    hooks = registry(("before_turn", record), ("on_final_response", record), ("after_run", record))
    api = ScriptedApi(["first", "second"])
    messages = [{"role": "user", "content": "go"}]
    events, text = drain(call_llm(messages, "m", api, hooks=hooks, steer=steer))
    assert text == "second"
    assert order == ["before_turn", "before_turn", "on_final_response", "after_run"]
    roles = []
    for message in messages:
        roles.append(message["role"])
    assert roles == ["user", "assistant", "user", "assistant"]
    assert messages[1] == {"role": "assistant", "content": "first"}
    assert messages[2] == {"role": "user", "content": "one more thing"}
    assert_transcript(messages)


def test_max_steps_raises_after_the_last_allowed_turn_hooks():
    order = []

    def record(ctx):
        order.append(str(ctx.event))

    hooks = registry(("before_turn", record), ("after_turn", record), ("after_run", record))
    api = ScriptedApi([[("poke", "{}")], [("poke", "{}")], [("poke", "{}")], "never"])
    messages = [{"role": "user", "content": "go"}]
    with pytest.raises(MaxStepsReached):
        drain(call_llm(messages, "m", api, tools=TOOLS, tools_dispatch=make_dispatch([]),
                       hooks=hooks, max_steps=2))
    assert order == ["before_turn", "after_turn", "before_turn", "after_turn"]
    assert len(api.calls) == 2
    assert_transcript(messages, finished=False)


def test_finish_beats_a_pending_steer_and_max_steps():
    """a finish on the last allowed turn ends the run before max_steps could
    trip and before a pending steer could fold in."""
    pending = ["later"]

    def steer():
        out = list(pending)
        pending.clear()
        return out

    api = ScriptedApi([[("poke", "{}")], "never"])
    messages = [{"role": "user", "content": "go"}]
    events, text = drain(call_llm(messages, "m", api, tools=TOOLS, tools_dispatch=make_dispatch([]),
                                  hooks=registry(("after_turn", returning(HookResult.finish("done")))),
                                  steer=steer, max_steps=1))
    assert text == "done"
    assert len(api.calls) == 1
    assert_transcript(messages)
    assert pending == []                   # drained before the first call
    roles = []
    for message in messages:
        roles.append(message["role"])
    assert roles == ["user", "user", "assistant", "tool", "assistant"]
