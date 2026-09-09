"""Tests for the draft-prompt primitives a `:`-command reaches through
ctx.screen - Screen.get_input / set_input on the unsent text in the input box.
The real Screen needs a tty, so the round-trip is covered on an instance built
without __init__."""
from cai.screen import Screen


def test_screen_input_round_trips_with_cursor_at_end():
    screen = Screen.__new__(Screen)
    screen._input_buf = list("hello")
    screen._cursor_pos = 2
    assert screen.get_input() == "hello"
    screen.set_input("a\nlonger draft")
    assert screen._input_buf == list("a\nlonger draft")
    assert screen._cursor_pos == len("a\nlonger draft")
    screen.set_input("")
    assert screen.get_input() == ""
    assert screen._cursor_pos == 0
