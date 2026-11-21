"""An overlay must not leave the alternate screen when it closes: the main TUI
lives there too, so an ALT_EXIT would flash the terminal's primary buffer (the
shell's scrollback) for a frame before the screen's restore repaints."""
import os
import pty
import shutil
import threading

from cai.screen.ansi import ALT_EXIT, KEY_ESC
from cai.screen.overlays.palette import prompt_palette_overlay


class _Screen:
    def __init__(self, tty_fd):
        self._tty_fd = tty_fd
        size = shutil.get_terminal_size()
        self._rows = size.lines
        self._cols = size.columns


def test_palette_escape_stays_in_the_alternate_screen(capsys):
    master, slave = pty.openpty()
    closed = threading.Event()

    def press_escape():
        # press once the overlay is up, and again only if it is still open
        # (well apart - two ESCs close together read as one Alt-ESC chord).
        delay = 0.2
        while not closed.wait(delay):
            os.write(master, KEY_ESC.encode())
            delay = 1.0

    presser = threading.Thread(target=press_escape, daemon=True)
    presser.start()
    try:
        picked = prompt_palette_overlay(_Screen(slave), [("models", "switch the model")])
    finally:
        closed.set()
        presser.join(timeout=2)
        os.close(master)
        os.close(slave)
    assert picked is None
    assert ALT_EXIT not in capsys.readouterr().out


def test_keys_typed_before_an_overlay_opens_reach_it():
    # the overlay frame enters raw mode without discarding pending input: a
    # filter and Enter typed ahead of the palette pick the command.
    master, slave = pty.openpty()
    try:
        os.write(master, b"mod\r")
        picked = prompt_palette_overlay(_Screen(slave), [("models", "switch the model"),
                                                         ("tools", "enable / disable tools")])
    finally:
        os.close(master)
        os.close(slave)
    assert picked == "models"
