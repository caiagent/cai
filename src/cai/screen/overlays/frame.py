"""overlay frame: the terminal handling every full-screen overlay shares.

An overlay takes the keyboard from Screen.prompt() for a while: it puts the
terminal in raw mode, follows window resizes itself, reads keys in its own
loop, and hands everything back the way it found it. OverlayFrame is that
frame, so an overlay's function holds only its own drawing and keys:

    frame = OverlayFrame(screen)
    try:
        frame.enter()
        redraw()
        while True:
            if frame.resized():
                redraw()
            key = frame.read_key()
            if key is None:
                continue
            ...
    finally:
        frame.leave()

Raw mode is entered without discarding pending input, so keys typed just
before the overlay opened (a fast ':models' then the first letters of a
filter) reach it instead of being lost."""
import select
import shutil
import signal
import termios
import tty

from ..input import read_key


class OverlayFrame:
    def __init__(self, screen):
        self._screen = screen
        self._fd = screen._tty_fd
        # what leave() restores: taken here, before the overlay touches
        # anything.
        self._attrs = termios.tcgetattr(self._fd)
        self._handler = signal.getsignal(signal.SIGWINCH)
        self._resized = False

    def _on_resize(self, signum, frame):
        size = shutil.get_terminal_size()
        self._screen._rows = size.lines
        self._screen._cols = size.columns
        self._resized = True

    def enter(self):
        """follow resizes and take the keyboard in raw mode."""
        signal.signal(signal.SIGWINCH, self._on_resize)
        self.raw()

    def raw(self):
        """(back) into raw mode, keeping what was typed meanwhile - also the
        way back after something borrowed the terminal (an editor)."""
        tty.setraw(self._fd, termios.TCSADRAIN)

    def leave(self):
        """give the terminal mode and the resize handler back."""
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._attrs)
        signal.signal(signal.SIGWINCH, self._handler)

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.leave()
        return False

    def resized(self):
        """whether the window was resized since the last call; the screen's
        size is already the new one."""
        if not self._resized:
            return False
        self._resized = False
        return True

    def read_key(self, timeout=0.05):
        """the next key, or None when none arrived within `timeout` seconds
        - so the loop comes round to notice a resize or a background
        update."""
        readable, _, _ = select.select([self._fd], [], [], timeout)
        if not readable:
            return None
        return read_key(self._fd)
