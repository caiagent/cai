"""Tests for --disallowed-paths: the CLI publishes the denied roots as
CAI_DISALLOWED_PATHS, cai.safe_path refuses them wherever they sit (a deny
wins over the cwd, scratch and any allowed-paths grant), spawned tool
processes inherit the var through os.environ, and the python-tool jail
knows them as denied roots."""
import os

import pytest

import cai
from cai import cli
from cai import pytool_bootstrap
from cai.tools import ToolsRegistry


def test_safe_path_refuses_a_denied_subtree_of_cwd(tmp_path, monkeypatch):
    secret = tmp_path / "secret"
    secret.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(secret))
    with pytest.raises(ValueError, match="disallowed"):
        cai.safe_path("secret")
    with pytest.raises(ValueError, match="disallowed"):
        cai.safe_path("secret/key.pem")
    # a sibling that merely shares the denied dir's name prefix stays open
    assert cai.safe_path("secret-notes") == str(tmp_path / "secret-notes")
    assert cai.safe_path("open.txt") == str(tmp_path / "open.txt")


def test_deny_wins_over_an_allowed_grant(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    private = extra / "private"
    private.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(extra))
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(private))
    assert cai.safe_path(str(extra / "ok.txt")) == str(extra / "ok.txt")
    with pytest.raises(ValueError, match="disallowed"):
        cai.safe_path(str(private / "x"))


def test_deny_wins_over_scratch(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    hidden = scratch / "hidden"
    hidden.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("CAI_SCRATCH", str(scratch))
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(hidden))
    assert cai.safe_path("$CAI_SCRATCH/a") == str(scratch / "a")
    with pytest.raises(ValueError, match="disallowed"):
        cai.safe_path("$CAI_SCRATCH/hidden/a")


def test_a_denied_file_blocks_just_that_file(tmp_path, monkeypatch):
    denied = tmp_path / "denied.txt"
    denied.write_text("x")
    (tmp_path / "sibling.txt").write_text("y")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(denied))
    with pytest.raises(ValueError, match="disallowed"):
        cai.safe_path("denied.txt")
    assert cai.safe_path("sibling.txt") == str(tmp_path / "sibling.txt")
    assert cai.safe_path(".") == str(tmp_path)


def test_unset_var_denies_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_SCRATCH", raising=False)
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    assert cai.safe_path("anything") == str(tmp_path / "anything")


def test_publish_sets_env_from_comma_list(tmp_path, monkeypatch):
    one = tmp_path / "one"
    one.mkdir()
    two = tmp_path / "two.txt"
    two.write_text("a file entry works too")
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    assert cli._publish_disallowed_paths(f"{one}, {two}") is True
    expected = os.pathsep.join([str(one), str(two)])
    assert os.environ["CAI_DISALLOWED_PATHS"] == expected


def test_publish_resolves_relative_entries_against_cwd(tmp_path, monkeypatch):
    (tmp_path / "rel").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    assert cli._publish_disallowed_paths("rel") is True
    assert os.environ["CAI_DISALLOWED_PATHS"] == str(tmp_path / "rel")


def test_publish_rejects_a_missing_entry(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CAI_DISALLOWED_PATHS", raising=False)
    assert cli._publish_disallowed_paths(str(tmp_path / "nope")) is False
    assert "CAI_DISALLOWED_PATHS" not in os.environ
    assert "--disallowed-paths entry does not exist" in capsys.readouterr().err


def test_main_exits_on_bad_disallowed_paths(tmp_path):
    assert cli.main(["-p", "x", "--disallowed-paths", str(tmp_path / "nope")]) == 1


def test_spawned_fs_server_inherits_the_deny(tmp_path, monkeypatch):
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "key.pem").write_text("private bytes")
    (tmp_path / "open.txt").write_text("public bytes")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(secret))

    registry = ToolsRegistry()
    registry.select("fs__read_file")
    try:
        out = registry.dispatch("fs__read_file", {"file_path": "open.txt"})
        assert "public bytes" in out
        out = registry.dispatch("fs__read_file", {"file_path": "secret/key.pem"})
        assert "private bytes" not in out
        assert "disallowed" in out
    finally:
        registry.close()


def test_jail_denied_roots_follow_the_var(tmp_path, monkeypatch):
    secret = tmp_path / "secret"
    secret.mkdir()
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(secret))
    assert pytool_bootstrap.compute_denied_roots() == [str(secret)]
    monkeypatch.delenv("CAI_DISALLOWED_PATHS")
    assert pytool_bootstrap.compute_denied_roots() == []
