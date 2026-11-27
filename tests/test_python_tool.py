"""Tests for the builtin `python` tool: the managed venv, running a snippet, and
the tool_call() tool-proxy that dispatches the agent's own tools in-process
through the run's gates. The child is really spawned; for speed most tests point
the venv at the current interpreter, and one test exercises real venv creation.
What a snippet may reach is the business of the sandbox it runs behind - see
test_sandbox.py."""
import os
import sys
import time

import pytest

import cai
from cai import config, hooks, pytool
from cai.agent import Agent
from cai.environment import Environment
from cai.hooks import HookResult, HooksRegistry, RunGate


def _fast_venv(monkeypatch):
    """skip real venv creation - run the child under this interpreter."""
    monkeypatch.setattr(pytool, "ensure_venv", lambda: sys.executable)


def _run(agent, code, timeout=20):
    return agent.tools_registry.dispatch("python", {"code": code, "timeout": timeout})


def _agent_with_echo():
    @cai.tool
    def echo(x: int) -> str:
        """echo x back."""
        return f"echo:{x}"
    return Agent(model="m", api=object(), tools=["echo"])


# --- wiring ---------------------------------------------------------------

def test_python_registered_unselected_until_the_skill():
    agent = Agent(model="m", api=object())
    try:
        assert "python" in agent.tools_registry.names()
        assert "python" not in agent.tools
    finally:
        agent.close()


def test_the_python_skill_selects_the_tool():
    agent = Agent(model="m", api=object(), skills=["python"])
    try:
        assert "python" in agent.tools
        assert agent.skills == ["python"]
    finally:
        agent.close()


def test_there_is_one_python_skill():
    # the read-only twin is gone: what a snippet may do is the sandbox's call.
    assert "python" in Environment.default().available_skills()
    assert "python-read-only" not in Environment.default().available_skills()


# --- venv -----------------------------------------------------------------

def test_ensure_venv_creates_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path))
    python = pytool.ensure_venv()
    assert os.path.exists(python)
    assert pytool.ensure_venv() == python


def test_ensure_venv_rebuilds_a_broken_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path))
    python = pytool.ensure_venv()
    os.remove(python)
    assert pytool.ensure_venv() == python
    assert os.path.exists(python)


def test_ensure_venv_uses_a_user_supplied_python_venv(tmp_path, monkeypatch):
    # python_venv points at an existing env: ensure_venv just returns its
    # interpreter, never running `python -m venv`.
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path / "cfg"))
    env = tmp_path / "myenv"
    monkeypatch.setattr(config, "load_optional",
                        lambda key, default=None: str(env) if key == "python_venv" else default)
    monkeypatch.setattr(config, "venv_dir", lambda: str(env))
    built = pytool.subprocess.run([sys.executable, "-m", "venv", str(env)])
    assert built.returncode == 0

    real_run = pytool.subprocess.run
    calls = []

    def record(cmd, *a, **k):
        calls.append(cmd)
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(pytool.subprocess, "run", record)

    assert pytool.ensure_venv() == config.venv_python()
    for cmd in calls:
        assert "venv" not in cmd, f"ensure_venv rebuilt a user-supplied env: {cmd}"


def test_ensure_venv_refuses_to_rebuild_a_broken_python_venv(tmp_path, monkeypatch):
    # a user-supplied env that is missing/unusable raises, never gets wiped.
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "venv_dir", lambda: str(tmp_path / "gone"))
    monkeypatch.setattr(config, "load_optional",
                        lambda key, default=None: str(tmp_path / "gone") if key == "python_venv" else default)
    with pytest.raises(RuntimeError, match="python_venv"):
        pytool.ensure_venv()


# --- running ---------------------------------------------------------------

def test_output_and_exit_code(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print(2 ** 10)").strip() == "1024"
        assert _run(agent, "x = 1") == "(no output)"
        assert "[exit code 3]" in _run(agent, "import sys; sys.exit(3)")
    finally:
        agent.close()


def test_timeout(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "while True: pass", timeout=1) == "Error: run timed out after 1s"
    finally:
        agent.close()



def _alive(pid):
    """True while pid is a live (non-zombie) process."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return False
    state = data.rsplit(")", 1)[1].split()[0]
    return state != "Z"


def test_daemon_is_swept_after_the_run(tmp_path, monkeypatch):
    # the child leads its own session; the whole process group is killed when
    # the run ends, so a spawned daemon cannot outlive it.
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    probe = """import subprocess
p = subprocess.Popen(['sleep', '60'])
print(p.pid)
"""
    try:
        pid = int(_run(agent, probe).strip())
    finally:
        agent.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _alive(pid):
            break
        time.sleep(0.05)
    assert not _alive(pid)


# --- no sandbox -------------------------------------------------------------

def test_without_a_sandbox_the_snippet_is_a_plain_process(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAI_TEST_MARK", "inherited")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    agent = _agent_with_echo()
    try:
        assert Environment.default().sandbox_for("python") is None
        # nothing is confined: a write outside cwd and scratch just lands...
        _run(agent, f"open({str(outside)!r}, 'w').write('x')")
        assert outside.read_text() == "x"
        # ...the environment is inherited whole, scratch is announced...
        out = _run(agent, "import os\n"
                          "print(os.environ['CAI_TEST_MARK'])\n"
                          "print(os.environ['CAI_SCRATCH'] == os.environ['TMPDIR'])")
        assert out.split() == ["inherited", "True"]
        # ...and the agent's tools are callable from the snippet.
        assert _run(agent, "print(echo(x=3))").strip() == "echo:3"
    finally:
        agent.close()
        if outside.exists():
            outside.unlink()


# --- tool proxy -----------------------------------------------------------

def test_call_proxies_a_tool_and_only_print_reaches_output(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # the raw result is computed but NOT printed; only the derived value is
        out = _run(agent, "r = tool_call('echo', x=7)\nprint(r.split(':')[1])")
        assert out.strip() == "7"
        assert "echo:7" not in out
    finally:
        agent.close()


def test_signatures_lists_selected_tools_but_not_python():
    agent = _agent_with_echo()
    try:
        agent.tools_registry.select("python")
        sigs = agent.tools_registry.signatures()
        assert "echo(x)" in sigs
        assert "python" not in sigs   # hidden: it cannot call itself
    finally:
        agent.close()


def test_skill_prompt_fills_the_tools_slot():
    from cai.skills import SkillsRegistry
    from cai.tools import ToolsRegistry

    @cai.tool
    def echo(x: int) -> str:
        """echo x back."""
        return f"echo:{x}"
    registry = ToolsRegistry.for_tools(["echo"])
    skills = SkillsRegistry(registry)
    skills._skills["s"] = type("S", (), {"body": "tools:\n{{tools}}"})()
    assert "echo(x)" in skills.system_prompt
    assert "{{tools}}" not in skills.system_prompt


def test_selected_tools_are_callable_by_name(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # the tool is in the snippet's namespace as a plain function - no
        # tool_call() needed; only the derived value is printed
        out = _run(agent, "print(echo(x=5).split(':')[1])")
        assert out.strip() == "5"
    finally:
        agent.close()


def test_python_is_not_offered_as_a_callable(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # python must not be a proxy in the namespace (it cannot call itself)
        assert "NameError" in _run(agent, "print(python)")
    finally:
        agent.close()


def test_call_recursion_and_unavailable_guards(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert "cannot call itself" in _run(agent, "print(tool_call('python', code='x'))")
        assert "not available" in _run(agent, "print(tool_call('secret'))")
    finally:
        agent.close()


def test_inner_call_is_gated_by_before_tool_call(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()

    def veto(ctx):
        return HookResult.veto()
    registry = HooksRegistry()
    registry.register("before_tool_call", veto)
    gate = RunGate(hooks=registry,
                   dispatch=agent.tools_registry.dispatch,
                   model="m",
                   config=None,
                   ui=hooks.NULL_UI,
                   messages=[],
                   usage=None,
                   hooks_data={})
    token = hooks.set_gate(gate)
    try:
        out = _run(agent, "print(tool_call('echo', x=1))")
        assert "aborted by a before_tool_call hook" in out
    finally:
        hooks.reset_gate(token)
        agent.close()


def test_inner_call_without_a_gate_still_dispatches(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # no run gate published (SDK-style direct use): confined dispatch, ungated
        assert hooks.current_gate() is None
        assert _run(agent, "print(tool_call('echo', x=9))").strip() == "echo:9"
    finally:
        agent.close()


def test_install_runs_pip_in_the_managed_venv(monkeypatch):
    monkeypatch.setattr(pytool, "ensure_venv", lambda: "/venv/bin/python")
    recorded = []

    class FakeResult:
        returncode = 3

    def fake_run(cmd):
        recorded.append(cmd)
        return FakeResult()
    monkeypatch.setattr(pytool.subprocess, "run", fake_run)

    assert pytool.install(["requests", "numpy>=2"]) == 3
    assert recorded == [["/venv/bin/python", "-m", "pip", "install",
                         "requests", "numpy>=2"]]


def test_uninstall_runs_pip_without_prompting(monkeypatch):
    monkeypatch.setattr(pytool, "ensure_venv", lambda: "/venv/bin/python")
    recorded = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd):
        recorded.append(cmd)
        return FakeResult()
    monkeypatch.setattr(pytool.subprocess, "run", fake_run)

    assert pytool.uninstall(["requests"]) == 0
    assert recorded == [["/venv/bin/python", "-m", "pip", "uninstall", "-y",
                         "requests"]]


def test_cli_python_subcommands_dispatch(monkeypatch):
    from cai import cli
    recorded = []

    def fake_install(packages):
        recorded.append(("install", packages))
        return 0

    def fake_uninstall(packages):
        recorded.append(("uninstall", packages))
        return 0

    def fake_list_packages():
        recorded.append(("list-packages", None))
        return 0
    monkeypatch.setattr(pytool, "install", fake_install)
    monkeypatch.setattr(pytool, "uninstall", fake_uninstall)
    monkeypatch.setattr(pytool, "list_packages", fake_list_packages)

    assert cli.main(["python", "install", "requests"]) == 0
    assert cli.main(["python", "uninstall", "requests"]) == 0
    assert cli.main(["python", "list-packages"]) == 0
    assert recorded == [("install", ["requests"]),
                        ("uninstall", ["requests"]),
                        ("list-packages", None)]
