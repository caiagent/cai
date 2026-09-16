"""Tests for the inline ask elements (screen/ask.py) and the Screen flow
around them: a UI request becomes a live block in the conversation, insert
mode keys answer it in place, Esc leaves it pending, and the waiting request
gets the element's result."""
import threading

from cai.screen.ansi import (KEY_ENTER, KEY_ESC, KEY_CTRL_C, KEY_BACKSPACE, KEY_TAB,
                             ansi_strip)
from cai.screen.ask import ConfirmAsk, SelectAsk, MultiAsk, TextAsk
from cai.screen.buffer import ContentBuffer
from cai.screen.layout import Layout
from cai.screen.modes import ModeHandler
from cai.screen.screen import Screen
from cai.screen.state import Mode, TUIState

ENTER = KEY_ENTER[0]


def _plain(element, width=60):
    return [ansi_strip(line) for line in element.lines(width)]


def test_confirm_moves_picks_and_shows_answer():
    ask = ConfirmAsk("Allow?", body="rm -rf /")
    assert _plain(ask)[:2] == ["? Allow?", "  rm -rf /"]
    assert ask.handle_key(KEY_TAB) is False
    assert ask.choice == 0
    assert ask.handle_key(ENTER) is True
    assert ask.result is True
    assert _plain(ask)[-1] == "  -> yes"
    assert ConfirmAsk("a").handle_key("n") is True
    cancelled = ConfirmAsk("a", default=True)
    cancelled.handle_key(KEY_CTRL_C)
    assert cancelled.result is True


def test_select_navigates_and_picks():
    ask = SelectAsk("Pick", ["a", "b"])
    ask.handle_key("j")
    ask.handle_key("j")
    assert ask.index == 1
    assert ask.cursor == (0, 0)
    _plain(ask)
    assert ask.cursor == (2, 2)
    assert ask.handle_key(ENTER) is True
    assert ask.result == "b"
    assert _plain(ask) == ["? Pick", "  -> b"]
    cancelled = SelectAsk("Pick", ["a"])
    cancelled.handle_key(KEY_CTRL_C)
    assert cancelled.result is None
    assert _plain(cancelled)[-1] == "  -> (cancelled)"


def test_multi_toggles_searches_and_answers_in_order():
    ask = MultiAsk("Which?", ["a", "b", "c"], default=["c"])
    ask.handle_key(" ")
    assert ask.handle_key("/") is False
    assert ask.searching is True
    ask.handle_key(KEY_ESC)
    assert ask.searching is False
    assert ask.handle_key(ENTER) is True
    assert ask.result == ["a", "c"]
    assert _plain(ask)[-1] == "  -> a, c"
    rows = _plain(MultiAsk("q", ["long option", "b"]))
    assert rows[1].startswith("  [ ] long option")
    cancelled = MultiAsk("q", ["a"])
    cancelled.handle_key(KEY_CTRL_C)
    assert cancelled.result is None


def test_text_types_deletes_and_defaults():
    ask = TextAsk("Name?", default="anon")
    for ch in "bob":
        ask.handle_key(ch)
    ask.handle_key(KEY_BACKSPACE)
    assert _plain(ask)[-1] == "  > bo"
    assert ask.cursor == (1, 6)
    assert ask.handle_key(ENTER) is True
    assert ask.result == "bo"
    empty = TextAsk("Name?", default="anon")
    empty.handle_key(ENTER)
    assert empty.result == "anon"
    assert _plain(empty)[-1] == "  -> anon"
    secret = TextAsk("pw", secret=True)
    secret.handle_key("x")
    assert _plain(secret)[-1] == "  > *"


def test_buffer_replace_segment_in_the_middle():
    buffer = ContentBuffer(80)
    buffer.append_text("one\n")
    buffer.append_text("two\n")
    buffer.append_text("three\n")
    assert buffer.segment_count() == 3
    assert buffer.segment_start(1) == 1
    buffer.replace_segment(1, "2a\n2b\n")
    assert [ansi_strip(l) for l in buffer._lines] == ["one", "2a", "2b", "three"]
    assert buffer.segment_start(2) == 3


class FakeScreen(Screen):
    """a Screen without a terminal: the buffer/state/lock plumbing the live
    element flow touches, with painting stubbed out."""

    def __init__(self):
        self._cols = 80
        self._rows = 24
        self._buffer = ContentBuffer(80)
        self._layout = Layout(24, 80)
        self._state = TUIState()
        self._state.mode = Mode.INSERT
        self._render_lock = threading.RLock()
        self._req_lock = threading.Lock()
        self._req_pending = None
        self._live = None
        self._live_seg = -1
        self._live_request = None
        self._live_nested = False
        self._prompt_abort = False
        self._closed = False
        self._current_kind = None
        self._new_content_below = False
        self._current_prompt_msg = '> '
        self._input_buf = []
        self._cursor_pos = 0
        self._resize_pending = False
        self._focus_stack = ['main']
        self._write_pending = False
        self._last_render_time = 0.0
        self._in_prompt = False
        self.frames = 0

    def _refresh_all(self):
        self.frames += 1

    def _place_live_cursor(self):
        pass

    def _refresh_content(self):
        pass

    def _refresh_status(self):
        pass


def _plain_lines(screen):
    return [ansi_strip(l) for l in screen._buffer._lines]


def test_worker_request_becomes_live_block_and_keys_answer_it():
    screen = FakeScreen()
    screen.write("hello\n", kind=Screen.LLM)
    results = {}

    def worker():
        results["value"] = screen.submit_request({"kind": "select", "title": "Pick", "options": ["a", "b"]})

    thread = threading.Thread(target=worker)
    thread.start()
    while screen._req_pending is None:
        pass
    screen._service_requests()
    assert screen._live is not None
    assert _plain_lines(screen)[-3:] == ["▌ ? Pick", "▌   a ", "▌   b "]

    modes = ModeHandler()
    modes.handle_key("j", screen._state, screen)
    assert screen._live is not None
    assert screen._live.index == 1
    modes.handle_key(ENTER, screen._state, screen)
    thread.join(timeout=2)
    assert results["value"] == "b"
    assert screen._live is None
    assert _plain_lines(screen)[-2:] == ["▌ ? Pick", "▌   -> b"]
    assert screen.frames >= 3


def test_esc_leaves_the_element_pending_and_i_returns():
    screen = FakeScreen()
    screen._req_pending = ({"kind": "text", "title": "Name?"}, threading.Event(), {})
    screen._service_requests()
    modes = ModeHandler()
    modes.handle_key(KEY_ESC, screen._state, screen)
    assert screen._state.mode == Mode.NORMAL
    assert screen._live is not None
    modes.handle_key("i", screen._state, screen)
    assert screen._state.mode == Mode.INSERT
    modes.handle_key("x", screen._state, screen)
    assert screen._live.buf == ["x"]
    modes.handle_key(ENTER, screen._state, screen)
    assert screen._live is None
    assert _plain_lines(screen)[-1] == "▌   -> x"


def test_live_cursor_line_counts_wrapped_body_rows():
    # a confirm detail longer than the width wraps into several buffer rows;
    # the parked cursor must land on the yes/no row, not `logical index`
    # rows below the segment start (which is a body row when it wrapped)
    screen = FakeScreen()
    screen.write("hello\n", kind=Screen.LLM)
    body = "x" * 200
    screen._req_pending = ({"kind": "confirm", "title": "Allow?", "body": body},
                           threading.Event(), {})
    screen._service_requests()
    lines = _plain_lines(screen)
    assert lines[-1] == "▌    yes    no "
    assert len(lines) > 4                       # the body really wrapped
    assert screen._live_cursor_line() == len(lines) - 1
    modes = ModeHandler()
    modes.handle_key(KEY_TAB, screen._state, screen)
    assert screen._live_cursor_line() == len(lines) - 1


def test_main_thread_request_nests_a_prompt_and_is_released():
    screen = FakeScreen()
    modes = ModeHandler()

    def fake_prompt(msg):
        screen._service_requests()
        modes.handle_key("y", screen._state, screen)
        assert screen._prompt_abort is True
        screen._prompt_abort = False

    screen.prompt = fake_prompt
    assert screen.submit_request({"kind": "confirm", "title": "ok?"}) is True
    assert screen._live is None
    assert screen._live_nested is False


def test_main_thread_request_abandoned_answers_default():
    screen = FakeScreen()

    def fake_prompt(msg):
        screen._service_requests()

    screen.prompt = fake_prompt
    assert screen.submit_request({"kind": "multiselect", "title": "q", "options": ["a"]}) is None
    assert screen._live is None
    assert _plain_lines(screen)[-1] == "▌   -> (cancelled)"


def test_close_wakes_a_live_request():
    screen = FakeScreen()
    done = threading.Event()
    screen._req_pending = ({"kind": "confirm", "title": "ok?"}, done, {})
    screen._service_requests()
    assert screen._live_request[0] is done
    screen._focus_stack = ['main']
    screen._tty_fd = None
    try:
        screen.close()
    except Exception:
        pass
    assert done.is_set()
