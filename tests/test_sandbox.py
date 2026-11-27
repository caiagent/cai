"""Tests for cai.sandbox: the command a tool process is started behind - the
python tool's child and the local MCP servers.

Most tests use a stand-in wrapper - a shell script that records the arguments
it was given and then runs its command with an EMPTY environment - so they
prove what cai hands a sandbox and that the tools survive one, without needing
a real sandbox installed. The MCP tests spawn a minimal stdio server written
here (no mcp package needed)."""
import os
import sys

import pytest

import cai
from cai import hooks, pytool
from cai.agent import Agent
from cai.environment import Environment, Extension
from cai.hooks import HookResult, HooksRegistry, RunGate
from cai.tools import ToolsRegistry, sandboxed_command


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


# --- the stand-in wrapper -------------------------------------------------------

def _wrapper(tmp_path, name="wrap"):
    """a stand-in sandbox command: appends every argument it was given to
    <name>.log (one per line, newlines inside one flattened, an ==end== line
    after each run), then runs the command that follows `--` with an EMPTY
    environment - the harshest thing a real sandbox does to it. returns
    (argv prefix, log path)."""
    log = tmp_path / (name + ".log")
    script = tmp_path / (name + ".sh")
    script.write_text("#!/bin/sh\n"
                      "for arg in \"$@\"; do\n"
                      f"    printf '%s' \"$arg\" | tr '\\n' ' ' >> '{log}'\n"
                      f"    echo >> '{log}'\n"
                      "done\n"
                      f"echo ==end== >> '{log}'\n"
                      "while [ \"$1\" != -- ]; do shift; done\n"
                      "shift\n"
                      "exec /usr/bin/env -i \"$@\"\n")
    script.chmod(0o755)
    return [str(script)], log


def _runs(log):
    """the argument lists the wrapper recorded, one per run."""
    if not log.exists():
        return []
    runs = []
    for chunk in log.read_text().split("==end==\n"):
        if not chunk: continue
        runs.append(chunk.rstrip("\n").split("\n"))
    return runs


# --- the python tool ------------------------------------------------------------

def test_the_python_tool_runs_behind_its_sandbox(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    prefix, log = _wrapper(tmp_path)
    cai.sandbox("python", prefix + ["--scratch", "{scratch}", "--cwd={cwd}", "--"])
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print(2 ** 10)").strip() == "1024"
        runs = _runs(log)
        assert len(runs) == 1
        args = runs[0]
        # the holes were filled with the agent's scratch dir and the cwd
        assert args[0] == "--scratch"
        assert args[1] == agent.scratch()
        assert args[2] == "--cwd=" + os.getcwd()
        assert args[3] == "--"
        # then env(1) with cai's variables, then the interpreter's command
        assert args[4] == "env"
        assert f"CAI_SCRATCH={agent.scratch()}" in args
        assert sys.executable in args
        assert args.index(f"CAI_SCRATCH={agent.scratch()}") < args.index(sys.executable)
    finally:
        agent.close()


def test_tool_calls_and_scratch_survive_a_sandbox_that_drops_the_environment(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOT_CAIS", "host")
    prefix, _ = _wrapper(tmp_path)
    cai.sandbox("python", prefix + ["--"])
    agent = _agent_with_echo()
    try:
        # the wrapper emptied the environment: what the host had is gone...
        assert _run(agent, "import os; print('NOT_CAIS' in os.environ)").strip() == "False"
        # ...but the tool channel still works, both spellings...
        assert _run(agent, "print(echo(x=5))").strip() == "echo:5"
        assert _run(agent, "print(tool_call('echo', x=6))").strip() == "echo:6"
        # ...and scratch is still announced, as CAI_SCRATCH and as TMPDIR.
        out = _run(agent, "import os, tempfile\n"
                          "print(os.environ['CAI_SCRATCH'])\n"
                          "print(tempfile.gettempdir())")
        assert out.split() == [agent.scratch(), agent.scratch()]
        # exit codes and stdin (the snippet itself) pass through too.
        assert "[exit code 3]" in _run(agent, "import sys; sys.exit(3)")
    finally:
        agent.close()


def test_a_sandboxed_inner_call_is_still_gated(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    prefix, _ = _wrapper(tmp_path)
    cai.sandbox("python", prefix + ["--"])
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


def test_the_star_sandbox_covers_python_unless_it_has_its_own(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    star, star_log = _wrapper(tmp_path, "star")
    own, own_log = _wrapper(tmp_path, "own")
    agent = _agent_with_echo()
    try:
        cai.sandbox("*", star + ["--"])
        assert _run(agent, "print(1)").strip() == "1"
        assert len(_runs(star_log)) == 1

        # its own registration wins over the default...
        cai.sandbox("python", own + ["--"])
        assert _run(agent, "print(2)").strip() == "2"
        assert len(_runs(star_log)) == 1
        assert len(_runs(own_log)) == 1

        # ...and None takes it out of the default altogether.
        cai.sandbox("python", None)
        assert _run(agent, "print(3)").strip() == "3"
        assert len(_runs(star_log)) == 1
        assert len(_runs(own_log)) == 1
    finally:
        agent.close()


def test_a_sandbox_registered_mid_session_applies_to_the_next_call(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    prefix, log = _wrapper(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print(1)").strip() == "1"
        assert _runs(log) == []
        cai.sandbox("python", prefix + ["--"])
        assert _run(agent, "print(2)").strip() == "2"
        assert len(_runs(log)) == 1
    finally:
        agent.close()


def test_a_sandbox_command_that_does_not_exist_is_an_error_result(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    cai.sandbox("python", [str(tmp_path / "no-such-sandbox"), "--"])
    agent = _agent_with_echo()
    before = set(os.listdir("/proc/self/fd"))
    try:
        out = _run(agent, "print(1)")
        assert out.startswith("Error:")
        assert "no-such-sandbox" in out
        # no pipe leaked on the failed spawn.
        assert set(os.listdir("/proc/self/fd")) == before
    finally:
        agent.close()


def test_a_sandbox_that_refuses_to_start_fails_the_call(tmp_path, monkeypatch):
    # a wrapper that exits non-zero without running its command: the result
    # carries its message and exit code, and the snippet never ran.
    _fast_venv(monkeypatch)
    marker = tmp_path / "ran"
    refuse = tmp_path / "refuse.sh"
    refuse.write_text("#!/bin/sh\necho 'refused: no such profile' >&2\nexit 2\n")
    refuse.chmod(0o755)
    cai.sandbox("python", [str(refuse)])
    agent = _agent_with_echo()
    try:
        out = _run(agent, f"open({str(marker)!r}, 'w').write('x')")
        assert "refused: no such profile" in out
        assert "[exit code 2]" in out
        assert not marker.exists()
    finally:
        agent.close()


def test_timeout_kills_a_sandboxed_run(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    prefix, _ = _wrapper(tmp_path)
    cai.sandbox("python", prefix + ["--"])
    agent = _agent_with_echo()
    try:
        assert _run(agent, "while True: pass", timeout=1) == "Error: run timed out after 1s"
    finally:
        agent.close()


# --- MCP servers --------------------------------------------------------------

# a stdio MCP server with one tool, `probe`, that reports what the process
# sees: its CAI_ variables, one variable a declaration sets, and whether a
# host variable reached it.
_SERVER = """\
import json, os, sys
for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message: continue
    result = {}
    if message["method"] == "tools/list":
        schema = {"type": "object", "properties": {}}
        result["tools"] = [{"name": "probe", "description": "probe", "inputSchema": schema}]
    if message["method"] == "tools/call":
        seen = {}
        for key in ("CAI_SCRATCH", "CAI_AGENT", "CAI_ALLOWED_PATHS", "TOKEN", "NOT_CAIS"):
            seen[key] = os.environ.get(key)
        result["content"] = [{"type": "text", "text": json.dumps(seen)}]
    reply = {"jsonrpc": "2.0", "id": message["id"], "result": result}
    sys.stdout.write(json.dumps(reply) + "\\n")
    sys.stdout.flush()
"""


def _server(tmp_path):
    path = tmp_path / "probe_server.py"
    path.write_text(_SERVER)
    return str(path)


def _probe(registry, name):
    import json
    return json.loads(registry.dispatch(name + "__probe", {}))


def test_a_declared_server_runs_behind_its_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("NOT_CAIS", "host")
    monkeypatch.setenv("CAI_ALLOWED_PATHS", "/granted")
    prefix, log = _wrapper(tmp_path)
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)], env={"TOKEN": "t"})
    cai.sandbox("probe", prefix + ["--rw", "{scratch}", "--"])
    agent = Agent(model="m", api=object(), tools=["probe__probe"], name="worker")
    try:
        seen = _probe(agent.tools_registry, "probe")
        # the wrapper emptied the environment, yet cai's own variables and
        # the declared one arrived; the host's did not.
        assert seen["CAI_SCRATCH"] == agent.scratch()
        assert seen["CAI_AGENT"] == "worker"
        assert seen["CAI_ALLOWED_PATHS"] == "/granted"
        assert seen["TOKEN"] == "t"
        assert seen["NOT_CAIS"] is None
        args = _runs(log)[-1]
        assert args[:3] == ["--rw", agent.scratch(), "--"]
    finally:
        agent.close()


def test_an_unsandboxed_server_inherits_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NOT_CAIS", "host")
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)])
    registry = ToolsRegistry.for_tools(["probe__probe"])
    try:
        assert _probe(registry, "probe")["NOT_CAIS"] == "host"
    finally:
        registry.close()


def test_a_server_file_in_an_mcps_dir_takes_the_star_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("NOT_CAIS", "host")
    mcps = tmp_path / "ext" / "mcps"
    os.makedirs(mcps)
    (mcps / "filed.py").write_text(_SERVER)
    env = Environment([Extension(name="ext", path=str(tmp_path / "ext"))])
    prefix, log = _wrapper(tmp_path)
    env.register_sandbox("*", prefix + ["--"])
    registry = ToolsRegistry(env)
    try:
        registry.select("filed__probe")
        assert _probe(registry, "filed")["NOT_CAIS"] is None
        assert len(_runs(log)) == 1
        assert os.path.join(str(mcps), "filed.py") in _runs(log)[0]
    finally:
        registry.close()


def test_none_takes_one_server_out_of_the_star_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("NOT_CAIS", "host")
    prefix, log = _wrapper(tmp_path)
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)])
    cai.sandbox("*", prefix + ["--"])
    cai.sandbox("probe", None)
    registry = ToolsRegistry.for_tools(["probe__probe"])
    try:
        assert _probe(registry, "probe")["NOT_CAIS"] == "host"
        assert _runs(log) == []
    finally:
        registry.close()


def test_listing_tools_spawns_behind_the_sandbox_with_a_throwaway_scratch(tmp_path):
    prefix, log = _wrapper(tmp_path)
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)])
    cai.sandbox("probe", prefix + ["--rw", "{scratch}", "--"])
    assert "probe__probe" in Environment.default().available_tools()
    runs = _runs(log)
    assert len(runs) == 1
    # there was no agent: a scratch dir was made for the listing and removed.
    assert runs[0][0] == "--rw"
    assert runs[0][1]
    assert not os.path.exists(runs[0][1])


def test_a_respawned_server_is_sandboxed_again(tmp_path):
    prefix, log = _wrapper(tmp_path)
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)])
    cai.sandbox("probe", prefix + ["--"])
    registry = ToolsRegistry.for_tools(["probe__probe"])
    try:
        _probe(registry, "probe")
        server = registry._mcp_servers["probe"]
        server._abort()                      # what an interrupt does
        _probe(registry, "probe")            # respawns lazily
        assert len(_runs(log)) == 2
    finally:
        registry.close()


def test_the_python_skill_says_whether_snippets_are_sandboxed():
    agent = Agent(model="m", api=object(), skills=["python"])
    try:
        plain = agent.system_prompt
        assert "- No sandbox: a snippet runs as an ordinary process of the user's." in plain
        cai.sandbox("python", ["wrap", "-p", "private-profile", "--"])
        boxed = agent.system_prompt
        assert "runs inside a\n  sandbox the user set up" in boxed
        assert "No sandbox" not in boxed
        # the model is told that there is a sandbox, never the command.
        assert "private-profile" not in boxed
        for prompt in (plain, boxed):
            assert os.path.realpath(os.getcwd()) in prompt
            assert "{{" not in prompt
    finally:
        agent.close()


# --- registration and the command line ------------------------------------------

def test_scratch_hole_without_a_scratch_dir_is_refused():
    with pytest.raises(ValueError, match="scratch"):
        sandboxed_command(["wrap", "--rw", "{scratch}", "--"], ["prog"])


def test_sandboxed_command_layout(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CAI_"):
            monkeypatch.delenv(key)
    # no variables to carry: the command follows the prefix directly
    assert sandboxed_command(["wrap", "--"], ["prog", "a"]) == ["wrap", "--", "prog", "a"]
    # cai's own variables and the given ones ride through env(1), sorted;
    # a given variable wins over an inherited CAI_ one; other host variables
    # are not carried.
    monkeypatch.setenv("CAI_ALLOWED_PATHS", "/granted")
    monkeypatch.setenv("CAI_AGENT", "inherited")
    monkeypatch.setenv("HOME_SECRET", "x")
    argv = sandboxed_command(["wrap", "{scratch}/x", "--"],
                             ["prog"],
                             env={"CAI_AGENT": "given", "TOKEN": "t"},
                             scratch="/s")
    assert argv == ["wrap", "/s/x", "--",
                    "env", "CAI_AGENT=given", "CAI_ALLOWED_PATHS=/granted", "TOKEN=t",
                    "prog"]
    # a brace that is not one of the two holes is left alone
    assert sandboxed_command(["wrap", "{home}", "{}"], ["p"])[:3] == ["wrap", "{home}", "{}"]


def test_sandbox_registration_is_validated():
    with pytest.raises(ValueError, match="argv list"):
        cai.sandbox("python", "wrap --")
    with pytest.raises(ValueError, match="empty"):
        cai.sandbox("python", [])
    with pytest.raises(ValueError, match="string"):
        cai.sandbox("python", ["wrap", 3])
    with pytest.raises(ValueError, match="name"):
        cai.sandbox("", ["wrap"])
    assert Environment.default().sandboxes() == {}


def test_sandbox_lookup():
    env = Environment()
    assert env.sandbox_for("python") is None
    env.register_sandbox("*", ["a"])
    env.register_sandbox("fs", None)
    env.register_sandbox("python", ["b", "--"])
    assert env.sandbox_for("python") == ["b", "--"]
    assert env.sandbox_for("github") == ["a"]
    assert env.sandbox_for("fs") is None
    # a later registration wins, and the returned list is a copy
    env.register_sandbox("python", ["c"])
    env.sandbox_for("python").append("mutated")
    assert env.sandbox_for("python") == ["c"]


# --- a sandbox that would cover nothing ---------------------------------------

def test_a_sandbox_for_a_name_nothing_answers_to_is_refused(tmp_path):
    env = Environment()
    env.register_sandbox("pyhton", ["wrap", "--"])
    with pytest.raises(ValueError, match="'pyhton' names nothing cai starts"):
        env.check_sandboxes()
    # taking a name out of "*" is checked the same way: it is as misspelled.
    env = Environment()
    env.register_sandbox("*", ["wrap", "--"])
    env.register_sandbox("nosuch", None)
    with pytest.raises(ValueError, match="nosuch"):
        env.check_sandboxes()


def test_a_sandbox_for_a_function_tool_is_refused():
    env = Environment()

    def lookup(x: int) -> str:
        """look something up."""
        return str(x)
    env.register_tool(lookup)
    env.register_sandbox("lookup", ["wrap", "--"])
    with pytest.raises(ValueError, match="function tool.*cai's own process"):
        env.check_sandboxes()


def test_a_sandbox_for_a_remote_server_is_refused():
    env = Environment()
    env.register_server("far", url="https://example.invalid/mcp")
    env.register_sandbox("far", ["wrap", "--"])
    with pytest.raises(ValueError, match="remote MCP server"):
        env.check_sandboxes()


def test_a_sandbox_for_one_tool_of_a_server_names_the_server(tmp_path):
    env = Environment()
    env.register_server("probe", command=[sys.executable, _server(tmp_path)])
    env.register_sandbox("probe__probe", ["wrap", "--"])
    with pytest.raises(ValueError, match="register it for 'probe'"):
        env.check_sandboxes()


def test_every_sandbox_that_covers_nothing_is_named_at_once():
    env = Environment()
    env.register_sandbox("one", ["wrap"])
    env.register_sandbox("two", ["wrap"])
    with pytest.raises(ValueError) as raised:
        env.check_sandboxes()
    assert "'one'" in str(raised.value)
    assert "'two'" in str(raised.value)


def test_the_names_a_sandbox_can_cover_pass(tmp_path):
    mcps = tmp_path / "ext" / "mcps"
    mcps.mkdir(parents=True)
    (mcps / "filed.py").write_text(_SERVER)
    env = Environment([Extension("ext", str(tmp_path / "ext"))])
    env.register_server("declared", command=[sys.executable, _server(tmp_path)])
    env.register_sandbox("*", ["wrap", "--"])
    env.register_sandbox("python", ["wrap", "--"])
    env.register_sandbox("declared", ["wrap", "--"])
    env.register_sandbox("filed", None)
    env.check_sandboxes()
    assert env.process_kind("python") == "python"
    assert env.process_kind("declared") == "local"
    assert env.process_kind("filed") == "local"
    assert env.process_kind("nosuch") is None


def test_a_server_may_be_declared_after_its_sandbox(tmp_path):
    # registration takes any name; the check comes once all is registered.
    cai.sandbox("probe", ["wrap", "--"])
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)])
    Environment.default().check_sandboxes()


def test_an_agent_is_not_built_on_a_sandbox_that_covers_nothing():
    cai.sandbox("pyhton", ["wrap", "--"])
    with pytest.raises(ValueError, match="pyhton"):
        Agent(model="m", api=object())


def test_load_refuses_a_sandbox_that_covers_nothing(tmp_path, monkeypatch):
    init = tmp_path / "init.py"
    init.write_text("import cai\ncai.sandbox('pyhton', ['wrap', '--'])\n")
    from cai import environment
    monkeypatch.setattr(environment, "init_path", lambda: str(init))
    monkeypatch.setattr(environment, "list_extensions", lambda: [])
    with pytest.raises(ValueError, match="pyhton"):
        Environment().load()


# --- which process serves which tool ------------------------------------------

def test_tool_processes_group_the_selected_tools_with_their_sandbox(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    cai.mcp_server("probe", command=[sys.executable, _server(tmp_path)])
    cai.mcp_server("far", url="https://example.invalid/mcp")
    prefix, _log = _wrapper(tmp_path)
    cai.sandbox("probe", prefix + ["--rw", "{scratch}", "--"])
    agent = _agent_with_echo()
    try:
        agent.tools_registry.select("probe__probe")
        agent.tools_registry.select("python")
        processes = agent.get_tool_processes()
        assert sorted(processes.keys()) == ["probe", "python"]
        # the command as registered: holes are filled per spawn, not here.
        assert processes["probe"] == {"kind": "local",
                                      "sandbox": prefix + ["--rw", "{scratch}", "--"],
                                      "tools": ["probe__probe"]}
        assert processes["python"] == {"kind": "python", "sandbox": None, "tools": ["python"]}
        # a function tool runs in cai's process and is in no entry.
        assert "echo" in agent.get_selected_tools()
        cai.sandbox("*", ["wrap", "--"])
        assert agent.get_tool_processes()["python"]["sandbox"] == ["wrap", "--"]
    finally:
        agent.close()
