"""the agent's live selection, enforced at the model boundary and mutable from
hooks mid-run.

tools: only the SELECTED tools are offered to the model, and only they can be
dispatched - a call to any other registered tool is refused, never run.
skills: a skill brings its prompt body and its tools; switching skills switches
both, and nothing of a dropped skill survives. model, tools, skills and the
system prompt are all read right before every model call, so a hook that
changes them changes the very next call. and the system prompt is byte-stable
across calls when nothing changed - a prompt that drifts on its own breaks
provider-side caching.

every test proves the negative too: it makes the model call what must be
refused and checks the refusal, and it checks what must still work still works.
fully offline: a scripted api plays the model."""
import copy
import os
import textwrap

import pytest

import cai
from cai.agent import Agent
from cai.environment import Environment, Extension


# --------------------------------------------------------------------------
# fakes / helpers
# --------------------------------------------------------------------------

EXECUTED = []


@pytest.fixture(autouse=True)
def _reset_executed():
    EXECUTED.clear()
    yield
    EXECUTED.clear()


def alpha() -> str:
    """alpha."""
    EXECUTED.append("alpha")
    return "alpha!"


def beta() -> str:
    """beta."""
    EXECUTED.append("beta")
    return "beta!"


def gamma() -> str:
    """gamma."""
    EXECUTED.append("gamma")
    return "gamma!"


SKILLS = {}
SKILLS["reader"] = """\
    name: reader
    tools: alpha
    ---
    READER BODY
    tools now:
    {{tools}}
    """
SKILLS["writer"] = """\
    name: writer
    tools: beta, gamma
    ---
    WRITER BODY
    """
SKILLS["stack"] = """\
    name: stack
    skills: reader
    tools: gamma
    ---
    STACK BODY
    """
SKILLS["ticker"] = """\
    name: ticker
    ---
    TICKER {{tick}}
    """


def make_env(tmp_path):
    """a private Environment: one extension carrying the SKILLS files, and the
    three function tools registered under their bare names."""
    skills_dir = tmp_path / "ext" / "skills"
    os.makedirs(skills_dir, exist_ok=True)
    for name, text in SKILLS.items():
        with open(skills_dir / (name + ".md"), "w") as f:
            f.write(textwrap.dedent(text))
    env = Environment([Extension(name="ext", path=str(tmp_path / "ext"))])
    env.register_tool(alpha)
    env.register_tool(beta)
    env.register_tool(gamma)
    return env


def wire_calls(step, call_no):
    calls = []
    for i, name in enumerate(step):
        function = {}
        function["name"] = name
        function["arguments"] = "{}"
        call = {}
        call["id"] = f"c{call_no}_{i}"
        call["type"] = "function"
        call["function"] = function
        calls.append(call)
    return calls


class ScriptedApi:
    """plays the model from a script: a list entry that is a list of tool names
    is a tool turn calling them all; a str is the final text. records what each
    call was made with - model, offered tool names, system prompt, messages."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, model, tools=None, stream=False, **kwargs):
        record = {}
        record["model"] = model
        names = None
        if tools:
            names = []
            for tool in tools:
                names.append(tool["function"]["name"])
        record["tools"] = names
        system = None
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
        record["system"] = system
        record["messages"] = copy.deepcopy(messages)
        self.calls.append(record)
        step = self.script.pop(0)
        usage = {"prompt_tokens": 10 * len(self.calls)}
        if isinstance(step, str):
            result = (step, None, None, usage)
        else:
            result = (None, None, wire_calls(step, len(self.calls)), usage)
        if not stream:
            return result
        def gen():
            yield result
        return gen()

    def offered(self, call_no):
        return self.calls[call_no]["tools"]

    def system(self, call_no):
        return self.calls[call_no]["system"]


def tool_results(messages):
    """{tool_call_id: content} over every tool message."""
    out = {}
    for message in messages:
        if message.get("role") != "tool": continue
        out[message["tool_call_id"]] = message["content"]
    return out


def refused(messages, call_id):
    return tool_results(messages)[call_id].startswith("Error: unknown tool")


def make_agent(tmp_path, api, **kwargs):
    return Agent(model="m", api=api, env=make_env(tmp_path), **kwargs)


def run(agent, prompt="go"):
    return list(agent.run(prompt))


# --------------------------------------------------------------------------
# tools: only the selected ones are offered, only they dispatch
# --------------------------------------------------------------------------

def test_only_selected_tools_are_offered_and_dispatchable(tmp_path):
    api = ScriptedApi([["alpha", "gamma"], "done"])
    agent = make_agent(tmp_path, api, tools=[alpha, beta])
    assert agent.get_selected_tools() == ["alpha", "beta"]
    run(agent)
    assert api.offered(0) == ["alpha", "beta"]          # gamma is registered on the env, not offered
    assert EXECUTED == ["alpha"]                        # gamma never ran
    assert refused(agent.messages, "c1_1")
    assert tool_results(agent.messages)["c1_0"] == "alpha!"


def test_set_selected_tools_narrows_offer_and_dispatch(tmp_path):
    api = ScriptedApi([["alpha", "beta"], "done"])
    agent = make_agent(tmp_path, api, tools=[alpha, beta])
    agent.set_selected_tools(["alpha"])
    assert agent.get_selected_tools() == ["alpha"]
    run(agent)
    assert api.offered(0) == ["alpha"]
    assert EXECUTED == ["alpha"]
    assert refused(agent.messages, "c1_1")


def test_agent_bound_tools_are_registered_but_not_offered(tmp_path):
    """the env registers the sub-agent / python tools on every agent; they must
    not leak into the offer or the dispatch unless selected."""
    api = ScriptedApi([["launch_agent"], "done"])
    agent = make_agent(tmp_path, api, tools=[alpha])
    registered = agent.tools_registry.names()
    assert "launch_agent" in registered
    run(agent)
    assert api.offered(0) == ["alpha"]
    assert refused(agent.messages, "c1_0")


def test_no_tools_selected_offers_none_and_refuses_all(tmp_path):
    api = ScriptedApi([["alpha"], "done"])
    agent = make_agent(tmp_path, api)
    run(agent)
    assert api.offered(0) is None
    assert EXECUTED == []
    assert refused(agent.messages, "c1_0")


# --------------------------------------------------------------------------
# skills: prompt body + tools, switched together
# --------------------------------------------------------------------------

def test_a_skill_brings_its_body_and_only_its_tools(tmp_path):
    api = ScriptedApi([["alpha", "beta"], "done"])
    agent = make_agent(tmp_path, api, skills=["reader"])
    assert agent.get_selected_skills() == ["reader"]
    assert agent.get_selected_tools() == ["alpha"]
    run(agent)
    assert "READER BODY" in api.system(0)
    assert "WRITER BODY" not in api.system(0)
    assert api.offered(0) == ["alpha"]
    assert EXECUTED == ["alpha"]
    assert refused(agent.messages, "c1_1")


def test_set_selected_skills_swaps_body_and_tools(tmp_path):
    api = ScriptedApi([["alpha", "beta"], "done"])
    agent = make_agent(tmp_path, api, skills=["reader"])
    agent.set_selected_skills(["writer"])
    assert agent.get_selected_skills() == ["writer"]
    assert sorted(agent.get_selected_tools()) == ["beta", "gamma"]
    run(agent)
    assert "WRITER BODY" in api.system(0)
    assert "READER BODY" not in api.system(0)
    assert sorted(api.offered(0)) == ["beta", "gamma"]
    assert EXECUTED == ["beta"]
    assert refused(agent.messages, "c1_0")           # alpha came with reader, now gone


def test_clearing_skills_clears_body_and_tools(tmp_path):
    api = ScriptedApi([["alpha"], "done"])
    agent = make_agent(tmp_path, api, skills=["reader"])
    agent.set_selected_skills([])
    run(agent)
    assert api.system(0) is None
    assert api.offered(0) is None
    assert refused(agent.messages, "c1_0")


def test_a_skill_stack_keeps_its_foundation_when_the_top_is_removed(tmp_path):
    """stack builds on reader: activating stack activates reader too; removing
    stack drops stack's body and gamma, reader and alpha stay."""
    api = ScriptedApi([["alpha", "gamma"], "done"])
    agent = make_agent(tmp_path, api, skills=["stack"])
    assert agent.get_selected_skills() == ["stack", "reader"]
    agent.set_selected_skills(["reader"])
    run(agent)
    assert "READER BODY" in api.system(0)
    assert "STACK BODY" not in api.system(0)
    assert api.offered(0) == ["alpha"]
    assert EXECUTED == ["alpha"]
    assert refused(agent.messages, "c1_1")


def test_base_prompt_comes_before_the_skill_bodies(tmp_path):
    api = ScriptedApi(["done"])
    agent = make_agent(tmp_path, api, system_prompt="BASE", skills=["reader"])
    run(agent)
    system = api.system(0)
    assert system.startswith("BASE")
    assert system.index("BASE") < system.index("READER BODY")


# --------------------------------------------------------------------------
# hooks that add / remove tools mid-run
# --------------------------------------------------------------------------

def test_hook_adding_a_tool_makes_it_callable_and_keeps_the_old_ones(tmp_path):
    api = ScriptedApi([["alpha", "beta"], ["alpha", "beta"], "done"])

    def add_beta(ctx):
        agent = cai.current_agent()
        agent.set_selected_tools(agent.get_selected_tools() + ["beta"])

    agent = make_agent(tmp_path, api, tools=[alpha], hooks=[("after_turn", add_beta)])
    run(agent)
    assert api.offered(0) == ["alpha"]
    assert api.offered(1) == ["alpha", "beta"]
    results = tool_results(agent.messages)
    assert results["c1_0"] == "alpha!"
    assert results["c1_1"].startswith("Error: unknown tool")   # beta before the hook
    assert results["c2_0"] == "alpha!"                          # alpha still works
    assert results["c2_1"] == "beta!"                           # beta now works
    assert EXECUTED == ["alpha", "alpha", "beta"]


def test_hook_removing_a_tool_refuses_it_and_keeps_the_rest(tmp_path):
    api = ScriptedApi([["alpha", "beta"], ["alpha", "beta"], "done"])

    def drop_beta(ctx):
        agent = cai.current_agent()
        keep = []
        for name in agent.get_selected_tools():
            if name == "beta": continue
            keep.append(name)
        agent.set_selected_tools(keep)

    agent = make_agent(tmp_path, api, tools=[alpha, beta], hooks=[("after_turn", drop_beta)])
    run(agent)
    assert api.offered(0) == ["alpha", "beta"]
    assert api.offered(1) == ["alpha"]
    results = tool_results(agent.messages)
    assert results["c1_0"] == "alpha!"
    assert results["c1_1"] == "beta!"
    assert results["c2_0"] == "alpha!"
    assert results["c2_1"].startswith("Error: unknown tool")
    assert EXECUTED == ["alpha", "beta", "alpha"]


def test_hook_removing_a_tool_between_the_call_and_its_dispatch(tmp_path):
    """the model names beta; a before_tool_call hook parks it first. dispatch
    reads the selection at call time, so beta is refused - and alpha, named in
    the same turn, still runs."""
    api = ScriptedApi([["beta", "alpha"], "done"])

    def park_beta(ctx):
        if ctx.tool_call.name != "beta": return None
        cai.current_agent().set_selected_tools(["alpha"])
        return None

    agent = make_agent(tmp_path, api, tools=[alpha, beta], hooks=[("before_tool_call", park_beta)])
    run(agent)
    results = tool_results(agent.messages)
    assert results["c1_0"].startswith("Error: unknown tool")
    assert results["c1_1"] == "alpha!"
    assert EXECUTED == ["alpha"]


# --------------------------------------------------------------------------
# hooks that add / remove skills mid-run
# --------------------------------------------------------------------------

def test_hook_adding_a_skill_adds_its_body_and_tools(tmp_path):
    api = ScriptedApi([["alpha", "beta"], ["alpha", "beta"], "done"])

    def add_writer(ctx):
        agent = cai.current_agent()
        agent.set_selected_skills(agent.get_selected_skills() + ["writer"])

    agent = make_agent(tmp_path, api, skills=["reader"], hooks=[("after_turn", add_writer)])
    run(agent)
    assert "WRITER BODY" not in api.system(0)
    assert "READER BODY" in api.system(1)
    assert "WRITER BODY" in api.system(1)
    assert api.offered(0) == ["alpha"]
    assert sorted(api.offered(1)) == ["alpha", "beta", "gamma"]
    results = tool_results(agent.messages)
    assert results["c1_1"].startswith("Error: unknown tool")
    assert results["c2_0"] == "alpha!"
    assert results["c2_1"] == "beta!"


def test_hook_removing_a_skill_removes_its_body_and_tools_and_keeps_the_rest(tmp_path):
    api = ScriptedApi([["alpha", "beta"], ["alpha", "beta"], "done"])

    def drop_reader(ctx):
        cai.current_agent().set_selected_skills(["writer"])

    agent = make_agent(tmp_path, api, skills=["reader", "writer"],
                       hooks=[("after_turn", drop_reader)])
    run(agent)
    assert "READER BODY" in api.system(0)
    assert "READER BODY" not in api.system(1)
    assert "WRITER BODY" in api.system(1)
    assert sorted(api.offered(1)) == ["beta", "gamma"]
    results = tool_results(agent.messages)
    assert results["c1_0"] == "alpha!"
    assert results["c2_0"].startswith("Error: unknown tool")   # alpha left with reader
    assert results["c2_1"] == "beta!"                           # writer's beta stays
    assert EXECUTED == ["alpha", "beta", "beta"]


def test_hook_swapping_skills_in_before_turn_shapes_that_very_call(tmp_path):
    api = ScriptedApi([["beta"], "done"])

    def swap(ctx):
        if ctx.usage is not None: return None      # first call only
        cai.current_agent().set_selected_skills(["writer"])
        return None

    agent = make_agent(tmp_path, api, skills=["reader"], hooks=[("before_turn", swap)])
    run(agent)
    assert "WRITER BODY" in api.system(0)
    assert "READER BODY" not in api.system(0)
    assert sorted(api.offered(0)) == ["beta", "gamma"]
    assert tool_results(agent.messages)["c1_0"] == "beta!"


# --------------------------------------------------------------------------
# hooks that change the model and the base prompt mid-run
# --------------------------------------------------------------------------

def test_hook_changing_the_model_changes_the_next_call_only(tmp_path):
    api = ScriptedApi([["alpha"], ["alpha"], "done"])

    def switch(ctx):
        if len(api.calls) != 1: return None
        cai.current_agent().set_model("m2")
        return None

    agent = make_agent(tmp_path, api, tools=[alpha], hooks=[("after_turn", switch)])
    run(agent)
    models = []
    for record in api.calls:
        models.append(record["model"])
    assert models == ["m", "m2", "m2"]


def test_hook_changing_the_base_prompt_keeps_the_skill_body(tmp_path):
    api = ScriptedApi([["alpha"], "done"])

    def rewrite(ctx):
        cai.current_agent().set_system_prompt_base("TERSE")

    agent = make_agent(tmp_path, api, system_prompt="KIND", skills=["reader"],
                       hooks=[("after_turn", rewrite)])
    run(agent)
    assert api.system(0).startswith("KIND")
    assert api.system(1).startswith("TERSE")
    assert "READER BODY" in api.system(1)


# --------------------------------------------------------------------------
# the system prompt is stable unless something changed
# --------------------------------------------------------------------------

def test_system_prompt_is_byte_identical_across_calls_when_nothing_changed(tmp_path):
    api = ScriptedApi([["alpha"], ["alpha"], ["alpha"], "done"])
    agent = make_agent(tmp_path, api, system_prompt="BASE", skills=["reader"])
    run(agent)
    systems = []
    for record in api.calls:
        systems.append(record["system"])
    assert len(api.calls) == 4
    assert len(set(systems)) == 1


def test_system_prompt_is_stable_across_runs_too(tmp_path):
    api = ScriptedApi(["one", "two"])
    agent = make_agent(tmp_path, api, system_prompt="BASE", skills=["reader"])
    run(agent, "first")
    run(agent, "second")
    assert api.system(0) == api.system(1)


def test_constant_slot_keeps_the_prompt_stable_and_fills_once_per_call(tmp_path):
    api = ScriptedApi([["alpha"], ["alpha"], "done"])
    env = make_env(tmp_path)
    fills = []

    def tick(ctx):
        fills.append(1)
        return "same"
    env.register_slot(tick)

    agent = Agent(model="m", api=api, env=env, tools=[alpha], skills=["ticker"])
    run(agent)
    systems = []
    for record in api.calls:
        systems.append(record["system"])
    assert len(set(systems)) == 1
    assert "TICKER same" in systems[0]
    assert len(fills) == 3                          # once per model call, not per hook


def test_system_prompt_changes_exactly_when_the_selection_changes(tmp_path):
    """reader's body carries {{tools}}: parking beta after call 2 changes the
    prompt once, and it stays put afterwards."""
    api = ScriptedApi([["alpha"], ["alpha"], ["alpha"], ["alpha"], "done"])

    def park_after_second(ctx):
        if len(api.calls) != 2: return None
        cai.current_agent().set_selected_tools(["alpha"])
        return None

    agent = make_agent(tmp_path, api, tools=[beta], skills=["reader"],
                       hooks=[("after_turn", park_after_second)])
    run(agent)
    systems = []
    for record in api.calls:
        systems.append(record["system"])
    assert systems[0] == systems[1]
    assert systems[1] != systems[2]
    assert systems[2] == systems[3] == systems[4]
    assert "beta" in systems[0]
    assert "beta" not in systems[2]


def test_strict_path_keeps_the_prompt_stable_across_attempts(tmp_path):
    api = ScriptedApi(["nope", "still no", "ok"])
    agent = make_agent(tmp_path, api, system_prompt="BASE", skills=["reader"])
    run(agent, "go")
    assert len(api.calls) == 1
    api = ScriptedApi(["nope", "still no", "ok"])
    agent = make_agent(tmp_path, api, system_prompt="BASE", skills=["reader"])
    list(agent.run("go", strict_format="regex:^ok$"))
    systems = []
    for record in api.calls:
        systems.append(record["system"])
    assert len(systems) == 3
    assert len(set(systems)) == 1
    assert systems[0].startswith("BASE")
    assert "READER BODY" in systems[0]
    assert systems[0].endswith("^ok$")
