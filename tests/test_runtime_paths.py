"""Tests for changing the path policy at runtime: cai.paths.publish is the
one writer of CAI_ALLOWED_PATHS / CAI_DISALLOWED_PATHS, ToolsRegistry
.restart_servers respawns the long-lived MCP servers under the new env, and
Agent.set_paths / get_paths tie the two together."""
import os

import pytest

from cai import paths
from cai.agent import Agent
from cai.tools import ToolsRegistry


class NoApi:
    def close(self):
        pass


def test_publish_exports_resolved_roots(tmp_path, monkeypatch):
    one = tmp_path / "one"
    one.mkdir()
    two = tmp_path / "two.txt"
    two.write_text("x")
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    paths.publish("CAI_ALLOWED_PATHS", f"{one}, {two}")
    assert os.environ["CAI_ALLOWED_PATHS"] == os.pathsep.join([str(one), str(two)])
    assert paths.allowed_paths() == [str(one), str(two)]


def test_publish_empty_spec_unsets_the_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(tmp_path))
    paths.publish("CAI_DISALLOWED_PATHS", "")
    assert "CAI_DISALLOWED_PATHS" not in os.environ
    assert paths.disallowed_paths() == []


def test_publish_missing_entry_raises_and_keeps_the_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(tmp_path))
    with pytest.raises(ValueError, match="does not exist"):
        paths.publish("CAI_ALLOWED_PATHS", str(tmp_path / "nope"))
    assert os.environ["CAI_ALLOWED_PATHS"] == str(tmp_path)


def test_restart_servers_respawns_under_the_new_env(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "data.txt").write_text("granted bytes")
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)

    registry = ToolsRegistry()
    registry.select("fs__read_file")
    try:
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "outside working directory" in out
        # the running server snapshotted its env: a bare change is not seen
        paths.publish("CAI_ALLOWED_PATHS", str(extra))
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "outside working directory" in out
        registry.restart_servers()
        assert registry.selected() == ["fs__read_file"]
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "granted bytes" in out
    finally:
        registry.close()


def test_agent_set_paths_publishes_and_reports(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    agent = Agent(model="m", api=NoApi())
    try:
        assert agent.get_paths() == {"allowed": [], "disallowed": []}
        agent.set_paths(allowed=str(extra), disallowed=str(secret))
        assert agent.get_paths() == {"allowed": [str(extra)], "disallowed": [str(secret)]}
        # None leaves a side alone, an empty spec clears it
        agent.set_paths(disallowed="")
        assert agent.get_paths() == {"allowed": [str(extra)], "disallowed": []}
    finally:
        agent.close()


def test_agent_set_paths_missing_entry_changes_nothing(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    agent = Agent(model="m", api=NoApi())
    try:
        with pytest.raises(ValueError):
            agent.set_paths(allowed=str(extra), disallowed=str(tmp_path / "nope"))
        assert agent.get_paths() == {"allowed": [], "disallowed": []}
    finally:
        agent.close()


def test_agent_set_paths_reaches_a_running_fs_server(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "data.txt").write_text("granted bytes")
    monkeypatch.delenv("CAI_ALLOWED_PATHS", raising=False)
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    agent = Agent(model="m", api=NoApi(), tools=["fs__read_file"])
    try:
        registry = agent.tools_registry
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "outside working directory" in out
        agent.set_paths(allowed=str(extra))
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "granted bytes" in out
        agent.set_paths(disallowed=str(extra / "data.txt"))
        out = registry.dispatch("fs__read_file", {"file_path": str(extra / "data.txt")})
        assert "disallowed" in out
    finally:
        agent.close()
