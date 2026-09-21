"""before_turn + the live model: the loop fires before_turn ahead of every model
call and reads the model afterwards through the getter the Agent hands it, so a
hook that calls agent.set_model(...) picks the model for that very call. fully
offline: a fake api records the model id each call was made with."""
import logging

import cai
from cai.agent import Agent
from cai.hooks import HookEvent, HookResult, HooksRegistry
from cai.llm import call_llm


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

def tool_call(name, call_id="c1"):
    function = {}
    function["name"] = name
    function["arguments"] = "{}"
    call = {}
    call["id"] = call_id
    call["type"] = "function"
    call["function"] = function
    return call


class RecordingApi:
    """turn 1 asks for the `poke` tool, turn 2 answers 'final'; records the
    model id of every call and reports usage on each."""

    def __init__(self):
        self.models = []

    def chat(self, messages, model, **kwargs):
        self.models.append(model)
        n = len(self.models)
        def gen():
            if n == 1:
                yield (None, None, [tool_call("poke")], {"prompt_tokens": 10})
            else:
                yield ("final", None, None, {"prompt_tokens": 20})
        return gen()


def drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def registry(*pairs):
    reg = HooksRegistry()
    for event, fn in pairs:
        reg.register(event, fn)
    return reg


def run_loop(api, model, hooks):
    def dispatch(name, args):
        return "poked"
    messages = [{"role": "user", "content": "go"}]
    tools = [{"type": "function", "function": {"name": "poke"}}]
    return drain(call_llm(messages, model, api, tools=tools,
                          tools_dispatch=dispatch, hooks=hooks))


def poke() -> str:
    """poke."""
    return "poked"


# --------------------------------------------------------------------------
# the live model getter
# --------------------------------------------------------------------------

def test_a_plain_model_id_still_works():
    api = RecordingApi()
    run_loop(api, "m", None)
    assert api.models == ["m", "m"]


def test_a_model_getter_is_read_before_every_call():
    api = RecordingApi()
    box = {"model": "m1"}

    def current():
        return box["model"]

    def switch(ctx):
        box["model"] = "m2"

    run_loop(api, current, registry(("after_turn", switch)))
    assert api.models == ["m1", "m2"]


def test_agent_set_model_from_an_after_turn_hook_applies_to_the_next_call():
    api = RecordingApi()

    def switch(ctx):
        cai.current_agent().set_model("m2")

    agent = Agent(model="m1", api=api, tools=[poke], hooks=[("after_turn", switch)])
    events = list(agent.run("go"))
    assert api.models == ["m1", "m2"]
    assert agent.model == "m2"


# --------------------------------------------------------------------------
# before_turn
# --------------------------------------------------------------------------

def test_before_turn_fires_ahead_of_every_call_with_the_last_usage():
    api = RecordingApi()
    seen = []

    def watch(ctx):
        seen.append((ctx.event, ctx.model, ctx.usage, len(ctx.messages)))

    run_loop(api, "m", registry(("before_turn", watch)))
    assert seen == [("before_turn", "m", None, 1),
                    ("before_turn", "m", {"prompt_tokens": 10}, 3)]


def test_before_turn_can_route_the_first_call():
    api = RecordingApi()

    def route(ctx):
        if ctx.usage is None:
            cai.current_agent().set_model("cheap")
        else:
            cai.current_agent().set_model("strong")

    agent = Agent(model="m", api=api, tools=[poke], hooks=[("before_turn", route)])
    list(agent.run("go"))
    assert api.models == ["cheap", "strong"]


def test_before_turn_is_an_observer(caplog):
    api = RecordingApi()

    def wrong(ctx):
        return HookResult.finish("bye")

    with caplog.at_level(logging.WARNING, logger="cai"):
        events, text = run_loop(api, "m", registry(("before_turn", wrong)))
    assert text == "final"                    # nothing stopped
    assert len(api.models) == 2
    assert "before_turn does not support" in caplog.text


def test_before_turn_is_a_valid_decorator_event():
    assert HookEvent("before_turn") is HookEvent.BEFORE_TURN

    def observe(ctx):
        return None
    HooksRegistry().register("before_turn", observe)


# --------------------------------------------------------------------------
# the model is never copied: every read resolves it at its own moment
# --------------------------------------------------------------------------

def test_a_switch_inside_the_tool_batch_is_seen_by_the_next_read():
    """the model changes while a tool runs (a thread, a command): the api call
    that follows uses the new id, and the after_turn context already sees it."""
    api = RecordingApi()
    box = {"model": "m1"}
    seen = []

    def current():
        return box["model"]

    def dispatch(name, args):
        box["model"] = "m2"
        return "poked"

    def after_tool(ctx):
        seen.append(ctx.model)

    messages = [{"role": "user", "content": "go"}]
    tools = [{"type": "function", "function": {"name": "poke"}}]
    hooks = registry(("after_tool_call", after_tool), ("after_turn", after_tool))
    drain(call_llm(messages, current, api, tools=tools, tools_dispatch=dispatch, hooks=hooks))
    assert api.models == ["m1", "m2"]
    assert seen == ["m2", "m2"]


def test_the_run_gate_resolves_the_model_at_dispatch_time():
    from cai.hooks import HookContext, RunGate, gated_dispatch
    box = {"model": "m1"}
    seen = []

    def current():
        return box["model"]

    def before(ctx):
        seen.append(ctx.model)

    def dispatch(name, args):
        return "ok"

    gate = RunGate(hooks=registry(("before_tool_call", before)),
                   dispatch=dispatch, model=current, config=None,
                   ui=None, messages=[], usage=None, hooks_data=None)
    gated_dispatch(gate, "poke", {})
    box["model"] = "m2"
    gated_dispatch(gate, "poke", {})
    assert seen == ["m1", "m2"]


# --------------------------------------------------------------------------
# the tool list is live too
# --------------------------------------------------------------------------

class ToolRecordingApi:
    """turn 1 asks for `poke`, turn 2 answers; records the tools each call
    offered."""

    def __init__(self):
        self.tools = []

    def chat(self, messages, model, **kwargs):
        self.tools.append(kwargs.get("tools"))
        n = len(self.tools)
        def gen():
            if n == 1:
                yield (None, None, [tool_call("poke")], {})
            else:
                yield ("final", None, None, {})
        return gen()


def test_a_tools_getter_is_read_before_every_call():
    api = ToolRecordingApi()
    box = {"tools": [{"type": "function", "function": {"name": "poke"}}]}

    def current():
        return box["tools"]

    def drop(ctx):
        box["tools"] = []

    def dispatch(name, args):
        return "poked"

    messages = [{"role": "user", "content": "go"}]
    schemas = list(box["tools"])
    drain(call_llm(messages, "m", api, tools=current, tools_dispatch=dispatch,
                   hooks=registry(("after_turn", drop))))
    assert api.tools == [schemas, None]     # offered on call 1, gone by call 2


def test_a_plain_tools_list_still_works_and_empty_reads_as_none():
    api = ToolRecordingApi()

    def dispatch(name, args):
        return "poked"

    schemas = [{"type": "function", "function": {"name": "poke"}}]
    drain(call_llm([{"role": "user", "content": "go"}], "m", api, tools=schemas,
                   tools_dispatch=dispatch))
    assert api.tools == [schemas, schemas]
    api = ToolRecordingApi()
    drain(call_llm([{"role": "user", "content": "go"}], "m", api, tools=[]))
    assert api.tools[0] is None


def test_agent_set_selected_tools_from_a_hook_changes_the_next_call():
    api = ToolRecordingApi()

    def park(ctx):
        cai.current_agent().set_selected_tools([])

    agent = Agent(model="m", api=api, tools=[poke], hooks=[("after_turn", park)])
    list(agent.run("go"))
    assert api.tools[0] is not None
    assert api.tools[0][0]["function"]["name"] == "poke"
    assert api.tools[1] is None
    assert agent.get_selected_tools() == []


# --------------------------------------------------------------------------
# the system prompt is live too
# --------------------------------------------------------------------------

class PromptRecordingApi:
    """turn 1 asks for `poke`, turn 2 answers; records the system message (or
    None) each call carried. non-streaming too, for the strict path."""

    def __init__(self):
        self.systems = []

    def _record(self, messages):
        system = None
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
        self.systems.append(system)
        return len(self.systems)

    def chat(self, messages, model, stream=False, **kwargs):
        n = self._record(messages)
        if not stream:
            if n == 1: return (None, None, [tool_call("poke")], {})
            return ("ok", None, None, {})
        def gen():
            if n == 1:
                yield (None, None, [tool_call("poke")], {})
            else:
                yield ("ok", None, None, {})
        return gen()


def test_a_system_prompt_getter_is_read_before_every_call():
    api = PromptRecordingApi()
    box = {"prompt": "one"}

    def current():
        return box["prompt"]

    def switch(ctx):
        box["prompt"] = "two"

    def dispatch(name, args):
        return "poked"

    tools = [{"type": "function", "function": {"name": "poke"}}]
    drain(call_llm([{"role": "user", "content": "go"}], "m", api, tools=tools,
                   tools_dispatch=dispatch, system_prompt=current,
                   hooks=registry(("after_turn", switch))))
    assert api.systems == ["one", "two"]


def test_agent_set_system_prompt_base_from_a_hook_changes_the_next_call():
    api = PromptRecordingApi()

    def rewrite(ctx):
        cai.current_agent().set_system_prompt_base("be terse")

    agent = Agent(model="m", api=api, tools=[poke], system_prompt="be kind",
                  hooks=[("after_turn", rewrite)])
    list(agent.run("go"))
    assert api.systems == ["be kind", "be terse"]


def test_strict_path_keeps_the_prompt_live_under_the_guidance():
    api = PromptRecordingApi()

    def rewrite(ctx):
        cai.current_agent().set_system_prompt_base("be terse")

    agent = Agent(model="m", api=api, tools=[poke], system_prompt="be kind",
                  hooks=[("after_turn", rewrite)])
    list(agent.run("go", strict_format="regex:^ok$"))
    assert api.systems[0].startswith("be kind")
    assert api.systems[1].startswith("be terse")
    for system in api.systems:
        assert system.endswith("must match the regular expression pattern: ^ok$")
