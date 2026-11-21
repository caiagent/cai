"""Screen._open_tty: which terminal the TUI reads keys from.

stdin wins whenever it is a terminal, so a sandbox that passes a pty on fd 0
but has no reachable /dev/tty (no controlling terminal, node not bind-mounted)
still runs the TUI. /dev/tty is opened only behind a piped stdin, and a
terminal on stderr/stdout is the last resort before a clear error."""
import builtins
import errno
import io
import sys

import pytest

from cai.screen.screen import Screen


class _Stream(io.StringIO):
    def __init__(self, fd, tty):
        super().__init__()
        self._fd = fd
        self._tty = tty

    def fileno(self):
        return self._fd

    def isatty(self):
        return self._tty


class _TtyFile:
    def __init__(self):
        self.closed = False

    def fileno(self):
        return 99

    def close(self):
        self.closed = True


@pytest.fixture
def streams(monkeypatch):
    def set_streams(stdin, stdout, stderr):
        monkeypatch.setattr(sys, "stdin", _Stream(0, stdin))
        monkeypatch.setattr(sys, "stdout", _Stream(1, stdout))
        monkeypatch.setattr(sys, "stderr", _Stream(2, stderr))
    return set_streams


@pytest.fixture
def dev_tty(monkeypatch):
    """stub open('/dev/tty'): `works` decides whether it opens; `opened` counts
    the attempts so a test can assert /dev/tty was never touched."""
    state = {"works": True, "opened": 0, "file": _TtyFile()}
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/dev/tty":
            state["opened"] += 1
            if not state["works"]:
                raise OSError(errno.ENXIO, "No such device or address")
            return state["file"]
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return state


def test_stdin_terminal_is_used_without_touching_dev_tty(streams, dev_tty):
    streams(stdin=True, stdout=True, stderr=True)
    dev_tty["works"] = False
    f, fd = Screen._open_tty()
    assert (f, fd) == (None, 0)
    assert dev_tty["opened"] == 0


def test_piped_stdin_falls_back_to_dev_tty(streams, dev_tty):
    streams(stdin=False, stdout=True, stderr=True)
    f, fd = Screen._open_tty()
    assert f is dev_tty["file"]
    assert fd == 99
    assert dev_tty["opened"] == 1


def test_no_dev_tty_falls_back_to_a_terminal_on_stderr(streams, dev_tty):
    streams(stdin=False, stdout=False, stderr=True)
    dev_tty["works"] = False
    assert Screen._open_tty() == (None, 2)


def test_no_terminal_anywhere_is_a_clear_error(streams, dev_tty):
    streams(stdin=False, stdout=False, stderr=False)
    dev_tty["works"] = False
    with pytest.raises(OSError) as excinfo:
        Screen._open_tty()
    assert "stdin is not a terminal" in str(excinfo.value)
    assert "/dev/tty" in str(excinfo.value)
    assert excinfo.value.errno == errno.ENXIO


def test_close_leaves_an_inherited_fd_open(streams, dev_tty, monkeypatch):
    """teardown closes only a /dev/tty the screen opened itself - never the
    process's own stdin."""
    import termios
    import signal
    import cai.screen.screen as mod
    streams(stdin=False, stdout=True, stderr=True)
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: ["attrs"])
    monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: None)
    monkeypatch.setattr(signal, "signal", lambda sig, h: None)
    monkeypatch.setattr(Screen, "_load_history", classmethod(lambda cls: []))
    screen = Screen()
    assert screen._tty_fd == 99
    screen.close()
    assert dev_tty["file"].closed

    streams(stdin=True, stdout=True, stderr=True)
    dev_tty["works"] = False
    screen = Screen()
    assert screen._tty_file is None and screen._tty_fd == 0
    screen.close()   # nothing to close; must not raise


def test_quiet_mode_keeps_typed_ahead_keys_without_echoing_them():
    # between two prompt() calls the main thread can be busy for seconds (a
    # command loading tools); what is typed meanwhile must neither be echoed
    # over the screen nor line-edited - it waits for the next prompt().
    import os
    import pty
    import termios

    master, slave = pty.openpty()
    try:
        cooked = termios.tcgetattr(slave)
        quiet = Screen._quiet_from(cooked)
        assert cooked[3] & termios.ECHO and cooked[3] & termios.ICANON
        assert not quiet[3] & termios.ECHO and not quiet[3] & termios.ICANON
        assert quiet[3] & termios.ISIG                     # Ctrl-C still interrupts
        assert not quiet[0] & termios.ICRNL                # Enter stays '\r', as in raw mode
        termios.tcsetattr(slave, termios.TCSADRAIN, quiet)
        os.write(master, b"typed\r")
        assert os.read(slave, 64) == b"typed\r"            # no newline needed, nothing rewritten
    finally:
        os.close(master)
        os.close(slave)


def test_prompt_leaves_quiet_mode_unless_it_was_nested_in_a_raw_one():
    import termios

    screen = Screen.__new__(Screen)
    screen._quiet_attrs = ["quiet"]
    cooked = [0, 0, 0, termios.ISIG | termios.ECHO | termios.ICANON, 0, 0, []]
    raw = [0, 0, 0, 0, 0, 0, []]
    assert screen._attrs_after_prompt(cooked) == ["quiet"]
    assert screen._attrs_after_prompt(raw) is raw
