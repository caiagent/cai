"""the hook examples under examples/extensions load as bundles and behave, fully
offline: each is copied into a temp HOME's extensions dir, Environment.load()
imports it, and its hook functions are called with hand-built contexts."""
import os
import shutil

import pytest

import cai
from cai import config
from cai.environment import Environment, extensions_dir
from cai.hooks import HookContext, HookResult, ToolCall, reset_agent, set_agent
from cai.ui import BaseUI


EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples", "extensions")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


def load(name):
    """install one example bundle and return {fn.__name__: (event, fn)}."""
    shutil.copytree(os.path.join(EXAMPLES, name), os.path.join(extensions_dir(), name))
    env = Environment().load()
    hooks = {}
    for event, fn in env.hooks():
        hooks[fn.__name__] = (event, fn)
    return env, hooks


def call(name, args, arguments="{}"):
    return ToolCall(name=name, arguments=arguments, args=args, id="c1")


def ctx(event, messages=None, tool_call=None, ui=None, usage=None, content=None):
    if messages is None: messages = []
    if ui is None: ui = BaseUI()
    return HookContext(event=event, messages=messages, model="m", ui=ui,
                       usage=usage, tool_call=tool_call, content=content)


def assistant_calls(*calls):
    """an assistant message carrying tool calls (name, arguments)."""
    tool_calls = []
    for name, arguments in calls:
        tool_calls.append({"id": "c", "type": "function",
                           "function": {"name": name, "arguments": arguments}})
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


class ConfirmingUI(BaseUI):
    interactive = True

    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def confirm(self, message, *, default=False, detail=""):
        self.asked.append(message)
        return self.answer


class FakeAgent:
    def __init__(self, model="m", tools=None):
        self.model = model
        self.selected = list(tools or [])
        self.set_models = []

    def set_model(self, model):
        self.set_models.append(model)
        self.model = model

    def get_selected_tools(self):
        return list(self.selected)

    def set_selected_tools(self, names):
        self.selected = list(names)


# --------------------------------------------------------------------------
# veto
# --------------------------------------------------------------------------

def test_veto_loads_two_before_tool_call_hooks():
    env, hooks = load("veto")
    assert hooks["deny_destructive"][0] == "before_tool_call"
    assert hooks["protect_paths"][0] == "before_tool_call"


def test_veto_denies_destructive_and_lets_others_through():
    env, hooks = load("veto")
    deny = hooks["deny_destructive"][1]
    result = deny(ctx("before_tool_call", tool_call=call("fs__remove_file", {"path": "x"})))
    assert result.vetoed is True
    assert "fs__remove_file" in result.reason
    assert deny(ctx("before_tool_call", tool_call=call("fs__read_file", {"path": "x"}))) is None


def test_veto_protects_paths_via_confirm_or_vetoes_headless():
    env, hooks = load("veto")
    protect = hooks["protect_paths"][1]
    edit_env = call("fs__edit_file", {"path": "app/.env"})
    headless = protect(ctx("before_tool_call", tool_call=edit_env))
    assert headless.vetoed is True
    assert "no user is reachable" in headless.reason
    yes = ConfirmingUI(True)
    assert protect(ctx("before_tool_call", tool_call=edit_env, ui=yes)) is None
    assert len(yes.asked) == 1
    no = ConfirmingUI(False)
    assert protect(ctx("before_tool_call", tool_call=edit_env, ui=no)).vetoed is True
    plain = call("fs__edit_file", {"path": "app/main.py"})
    assert protect(ctx("before_tool_call", tool_call=plain, ui=no)) is None


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------

def test_judge_vetoes_above_threshold_and_fails_closed(monkeypatch):
    env, hooks = load("judge")
    judge = hooks["judge"][1]
    assert hooks["judge"][0] == "before_tool_call"
    seen = []

    def fake_decide(state, questions, model=None, api=None):
        seen.append(state)
        answers = {}
        answers["risky"] = {"type": "noul", "noul": 0.95}
        return answers, {}

    monkeypatch.setattr(cai, "decide", fake_decide, raising=False)
    messages = [{"role": "user", "content": "clean the build dir"}]
    risky = judge(ctx("before_tool_call", messages=messages,
                      tool_call=call("fs__remove_file", {"path": "build"})))
    assert risky.vetoed is True
    assert "0.95" in risky.reason
    assert seen[0]["tool"] == "fs__remove_file"
    assert seen[0]["last_user_message"] == "clean the build dir"

    def safe_decide(state, questions, model=None, api=None):
        return {"risky": {"type": "noul", "noul": 0.1}}, {}

    monkeypatch.setattr(cai, "decide", safe_decide, raising=False)
    assert judge(ctx("before_tool_call", tool_call=call("fs__edit_file", {"path": "a"}))) is None

    def broken_decide(state, questions, model=None, api=None):
        raise cai.ApiError("down")

    monkeypatch.setattr(cai, "decide", broken_decide, raising=False)
    closed = judge(ctx("before_tool_call", tool_call=call("fs__edit_file", {"path": "a"})))
    assert closed.vetoed is True
    assert "down" in closed.reason


def test_judge_skips_read_only_tools(monkeypatch):
    env, hooks = load("judge")
    judge = hooks["judge"][1]

    def never(state, questions, model=None, api=None):
        raise AssertionError("read-only tools must not be judged")

    monkeypatch.setattr(cai, "decide", never, raising=False)
    assert judge(ctx("before_tool_call", tool_call=call("fs__read_file", {"path": "a"}))) is None


# --------------------------------------------------------------------------
# stuck
# --------------------------------------------------------------------------

def test_stuck_finishes_on_a_repeated_call():
    env, hooks = load("stuck")
    stop = hooks["stop_when_stuck"][1]
    assert hooks["stop_when_stuck"][0] == "after_turn"
    messages = [{"role": "user", "content": "go"}]
    for _ in range(3):
        messages.append(assistant_calls(("fs__read_file", '{"path": "a"}')))
        messages.append({"role": "tool", "tool_call_id": "c", "content": "..."})
    result = stop(ctx("after_turn", messages=messages))
    assert result.stop is True
    assert "fs__read_file" in result.content


def test_stuck_ignores_varied_calls_and_counts_from_the_last_user_turn():
    env, hooks = load("stuck")
    stop = hooks["stop_when_stuck"][1]
    messages = [{"role": "user", "content": "go"},
                assistant_calls(("fs__read_file", '{"path": "a"}')),
                assistant_calls(("fs__read_file", '{"path": "b"}')),
                assistant_calls(("fs__read_file", '{"path": "a"}'))]
    assert stop(ctx("after_turn", messages=messages)) is None
    messages = [{"role": "user", "content": "go"},
                assistant_calls(("x", "{}")),
                assistant_calls(("x", "{}")),
                {"role": "user", "content": "again"},
                assistant_calls(("x", "{}"))]
    assert stop(ctx("after_turn", messages=messages)) is None


def test_stuck_finishes_at_the_turn_budget():
    env, hooks = load("stuck")
    budget = hooks["stop_at_budget"][1]
    messages = [{"role": "user", "content": "go"}]
    for i in range(24):
        messages.append(assistant_calls(("t", f'{{"i": {i}}}')))
    assert budget(ctx("after_turn", messages=messages)) is None
    messages.append(assistant_calls(("t", '{"i": 24}')))
    result = budget(ctx("after_turn", messages=messages))
    assert result.stop is True
    assert "25 tool turns" in result.content


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------

def test_route_switches_the_agent_model_by_prompt_size(monkeypatch):
    env, hooks = load("route")
    route = hooks["route"][1]
    assert hooks["route"][0] == "before_turn"
    assert "route" in env.commands()
    models = {}
    models["route_small_model"] = "small"
    models["route_large_model"] = "large"

    def load_optional(key, default=None):
        return models.get(key, default)

    monkeypatch.setattr(config, "load_optional", load_optional)
    agent = FakeAgent(model="m")
    token = set_agent(agent)
    try:
        route(ctx("before_turn", usage=None))
        assert agent.model == "small"
        route(ctx("before_turn", usage={"prompt_tokens": 1000}))
        assert agent.model == "small"
        route(ctx("before_turn", usage={"prompt_tokens": 70_000}))
        assert agent.model == "large"
    finally:
        reset_agent(token)
    assert agent.set_models == ["small", "large"]


def test_route_is_quiet_without_config_or_agent(monkeypatch):
    env, hooks = load("route")
    route = hooks["route"][1]
    assert route(ctx("before_turn")) is None       # no agent published

    def unset(key, default=None):
        return default

    monkeypatch.setattr(config, "load_optional", unset)
    agent = FakeAgent()
    token = set_agent(agent)
    try:
        route(ctx("before_turn", usage={"prompt_tokens": 70_000}))
    finally:
        reset_agent(token)
    assert agent.set_models == []


# --------------------------------------------------------------------------
# footer
# --------------------------------------------------------------------------

def test_footer_rewrites_the_answer():
    env, hooks = load("footer")
    footer = hooks["footer"][1]
    assert hooks["footer"][0] == "on_final_response"
    result = footer(ctx("on_final_response", content="done  \n", usage={"total_tokens": 42}))
    assert result == HookResult.answer("done\n\n-- m, 42 tokens")
    bare = footer(ctx("on_final_response", content="done  "))
    assert bare == HookResult.answer("done")


# --------------------------------------------------------------------------
# nudge
# --------------------------------------------------------------------------

def test_nudge_appends_a_reminder_every_eight_tool_turns():
    env, hooks = load("nudge")
    nudge = hooks["nudge"][1]
    assert hooks["nudge"][0] == "after_turn"
    messages = [{"role": "user", "content": "go"}]
    for i in range(7):
        messages.append(assistant_calls(("t", "{}")))
        assert nudge(ctx("after_turn", messages=messages)) is None
    assert len(messages) == 8
    messages.append(assistant_calls(("t", "{}")))
    nudge(ctx("after_turn", messages=messages))
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("[nudge] 8 tool turns")
    # the nudge itself does not reset the count: 8 more turns -> 16
    for i in range(8):
        messages.append(assistant_calls(("t", "{}")))
    nudge(ctx("after_turn", messages=messages))
    assert messages[-1]["content"].startswith("[nudge] 16 tool turns")


# --------------------------------------------------------------------------
# focus
# --------------------------------------------------------------------------

def test_focus_parks_and_restores_write_tools():
    env, hooks = load("focus")
    focus = hooks["focus"][1]
    assert hooks["focus"][0] == "before_turn"
    agent = FakeAgent(tools=["fs__read_file", "fs__edit_file", "fs__create_file"])
    token = set_agent(agent)
    try:
        focus(ctx("before_turn", messages=[{"role": "user", "content": "look around, read-only please"}]))
        assert agent.selected == ["fs__read_file"]
        focus(ctx("before_turn", messages=[{"role": "user", "content": "still read only"}]))
        assert agent.selected == ["fs__read_file"]
        focus(ctx("before_turn", messages=[{"role": "user", "content": "now fix it"}]))
        assert sorted(agent.selected) == ["fs__create_file", "fs__edit_file", "fs__read_file"]
    finally:
        reset_agent(token)


def test_focus_is_quiet_without_an_agent():
    env, hooks = load("focus")
    focus = hooks["focus"][1]
    assert focus(ctx("before_turn", messages=[{"role": "user", "content": "read-only"}])) is None
