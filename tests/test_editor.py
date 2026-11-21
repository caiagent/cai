"""The editor command line: $VISUAL / $EDITOR plus the flags each known
editor needs to wait, open read-only and land on a row / column."""
import pytest

from cai.screen import input as screen_input
from cai.screen.input import editor_argv, run_editor


@pytest.fixture(autouse=True)
def _no_editor_env(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)


def test_visual_wins_over_editor(monkeypatch):
    monkeypatch.setenv("EDITOR", "nano")
    monkeypatch.setenv("VISUAL", "vim")
    assert editor_argv("/f") == ["vim", "/f"]


def test_without_either_the_first_installed_fallback_is_used(monkeypatch):
    monkeypatch.setattr(screen_input.shutil, "which", lambda name: name == "vim")
    assert editor_argv("/f") == ["vim", "/f"]
    monkeypatch.setattr(screen_input.shutil, "which", lambda name: None)
    assert editor_argv("/f") == ["vi", "/f"]


def test_the_value_is_a_command_line_not_a_program_name(monkeypatch):
    monkeypatch.setenv("EDITOR", "'/opt/my editor/ed' --flag")
    assert editor_argv("/f") == ["/opt/my editor/ed", "--flag", "/f"]


def test_an_unknown_editor_just_gets_the_path(monkeypatch):
    monkeypatch.setenv("EDITOR", "myedit")
    assert editor_argv("/f", readonly=True, row=3, col=4) == ["myedit", "/f"]


@pytest.mark.parametrize("editor, expected", [
    ("vim", ["vim", "+3", "+normal! 4|", "/f"]),
    ("/usr/bin/nvim", ["/usr/bin/nvim", "+3", "+normal! 4|", "/f"]),
    ("nano", ["nano", "+3,4", "/f"]),
    ("emacs -nw", ["emacs", "-nw", "+3:4", "/f"]),
    ("hx", ["hx", "/f:3:4"]),
    ("code", ["code", "--wait", "--goto", "/f:3:4"]),
    ("subl", ["subl", "--wait", "/f:3:4"]),
    ("kate", ["kate", "--block", "-l", "3", "-c", "4", "/f"]),
])
def test_cursor_position_is_spelled_the_editors_way(monkeypatch, editor, expected):
    monkeypatch.setenv("EDITOR", editor)
    assert editor_argv("/f", row=3, col=4) == expected


def test_read_only_uses_the_editors_flag(monkeypatch):
    monkeypatch.setenv("EDITOR", "vim")
    assert editor_argv("/f", readonly=True) == ["vim", "-R", "/f"]
    monkeypatch.setenv("EDITOR", "nano")
    assert editor_argv("/f", readonly=True) == ["nano", "-v", "/f"]
    monkeypatch.setenv("EDITOR", "code")                   # has none: opened writable
    assert editor_argv("/f", readonly=True) == ["code", "--wait", "/f"]


@pytest.mark.parametrize("editor", ["code --wait", "code -w", "Code.exe -w"])
def test_a_wait_flag_the_user_gave_is_not_added_twice(monkeypatch, editor):
    monkeypatch.setenv("EDITOR", editor)
    argv = editor_argv("/f")
    assert argv[-1] == "/f" and len(argv) == 3


def test_a_window_editor_says_what_the_terminal_is_waiting_for(monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(screen_input.subprocess, "run", ran.append)
    monkeypatch.setenv("EDITOR", "code")
    assert run_editor("/f") is True
    assert ran == [["code", "--wait", "/f"]]
    assert "waiting for code" in capsys.readouterr().out
    monkeypatch.setenv("EDITOR", "vim")
    assert run_editor("/f") is True
    assert capsys.readouterr().out == ""


def test_an_editor_that_cannot_start_is_reported_not_raised(monkeypatch, capsys):
    def missing(argv):
        raise FileNotFoundError(2, "No such file or directory", argv[0])
    monkeypatch.setattr(screen_input.subprocess, "run", missing)
    monkeypatch.setattr(screen_input.os, "read", lambda fd, n: b"\n")
    monkeypatch.setattr(screen_input.sys.stdin, "fileno", lambda: 0, raising=False)
    monkeypatch.setenv("EDITOR", "nosuch")
    assert run_editor("/f") is False
    assert "could not run the editor 'nosuch'" in capsys.readouterr().out
