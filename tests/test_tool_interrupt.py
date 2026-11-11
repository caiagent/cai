"""Ctrl-C during a tool call: a stdio MCP tool call made with the run's
interrupt Event set returns promptly with an 'interrupted' result instead of
waiting the tool out, the server is told to cancel, and the next call on the
same server still works. the loop itself is untouched: it reads the result and
carries on to the next tool call."""
import os
import sys
import threading
import time

import cai
from cai.tools import ToolsRegistry, _POLL_TICK


_SLOW_SERVER = """
import time
from mcp.server.mcpserver import MCPServer
mcp = MCPServer(name="slow")

@mcp.tool()
def nap(seconds: float) -> str:
    time.sleep(seconds)
    return f"slept {seconds:g}"

@mcp.tool()
def ping() -> str:
    return "pong"

mcp.run()
"""


class _Agent:
    def __init__(self):
        self.name = "test"
        self.interrupt = threading.Event()

    def scratch(self):
        return None


def _registry(tmp_path):
    server = tmp_path / "slow.py"
    server.write_text(_SLOW_SERVER)
    cai.mcp_server("slow", command=[sys.executable, str(server)])
    agent = _Agent()
    registry = ToolsRegistry(agent=agent)
    registry.select("slow__nap")
    registry.select("slow__ping")
    return registry, agent


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid): return True
        time.sleep(0.05)
    return not _alive(pid)


def test_interrupt_kills_the_server_and_the_next_call_respawns_it(tmp_path):
    registry, agent = _registry(tmp_path)
    try:
        assert registry.dispatch("slow__ping", {}) == "pong"
        server = registry._mcp_servers["slow"]
        first_pid = server._process.pid

        threading.Timer(0.3, agent.interrupt.set).start()
        started = time.monotonic()
        out = registry.dispatch("slow__nap", {"seconds": 10})
        elapsed = time.monotonic() - started
        assert out == "Error: tool 'slow__nap' interrupted by user"
        # the sleep did not run out: the server was killed, not waited for.
        assert elapsed < 0.3 + _POLL_TICK * 5 + 1.0
        assert _wait_gone(first_pid)

        # the loop clears the flag before the next run; the next call brings a
        # fresh server up and works as before.
        agent.interrupt.clear()
        assert registry.dispatch("slow__ping", {}) == "pong"
        assert server._process.pid != first_pid
        assert registry.dispatch("slow__nap", {"seconds": 0.1}) == "slept 0.1"
    finally:
        registry.close()


_FORKING_SERVER = """
import subprocess
from mcp.server.mcpserver import MCPServer
mcp = MCPServer(name="forking")

@mcp.tool()
def spawn_sleeper(seconds: float) -> str:
    # what fs search really does: block on a child (rg) of its own
    proc = subprocess.Popen(["sleep", str(seconds)])
    with open("sleeper.pid", "w") as f:
        f.write(str(proc.pid))
    proc.wait()
    return "done"

mcp.run()
"""


def test_interrupt_kills_the_tools_own_children_too(tmp_path, monkeypatch):
    """the fs search blocks on an rg child; the kill must take the whole
    process tree, or the child grinds on after the server is gone."""
    monkeypatch.chdir(tmp_path)
    server = tmp_path / "forking.py"
    server.write_text(_FORKING_SERVER)
    cai.mcp_server("forking", command=[sys.executable, str(server)])
    agent = _Agent()
    registry = ToolsRegistry(agent=agent)
    registry.select("forking__spawn_sleeper")
    try:
        threading.Timer(0.5, agent.interrupt.set).start()
        out = registry.dispatch("forking__spawn_sleeper", {"seconds": 30})
        assert out == "Error: tool 'forking__spawn_sleeper' interrupted by user"
        child_pid = int((tmp_path / "sleeper.pid").read_text())
        assert _wait_gone(child_pid)
    finally:
        registry.close()


def test_a_set_interrupt_is_seen_before_the_call_blocks(tmp_path):
    registry, agent = _registry(tmp_path)
    try:
        agent.interrupt.set()
        started = time.monotonic()
        out = registry.dispatch("slow__nap", {"seconds": 10})
        assert out == "Error: tool 'slow__nap' interrupted by user"
        assert time.monotonic() - started < _POLL_TICK * 5
        # refused before the request went out: the server was not touched.
        assert registry._mcp_servers["slow"]._process.poll() is None
    finally:
        registry.close()


def test_a_registry_without_an_agent_still_blocks_to_completion(tmp_path):
    server = tmp_path / "slow.py"
    server.write_text(_SLOW_SERVER)
    cai.mcp_server("slow", command=[sys.executable, str(server)])
    registry = ToolsRegistry()
    registry.select("slow__nap")
    try:
        assert registry.dispatch("slow__nap", {"seconds": 0.2}) == "slept 0.2"
    finally:
        registry.close()
