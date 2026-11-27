"""arrow / page keys in the prompt input: moving between the prompt's lines,
and stepping through the prompt history only when nothing would be lost."""
from cai.screen.modes import ModeHandler


class _Layout:
    input_height = 3


class _Scr:
    def __init__(self, text, pos, history=()):
        self._input_buf = list(text)
        self._cursor_pos = pos
        self._history = list(history)
        self._history_idx = -1
        self._history_draft = []
        self._layout = _Layout()

    def _refresh_input(self):
        pass

    @property
    def text(self):
        return ''.join(self._input_buf)


def _up(scr): ModeHandler()._handle_insert_arrow_up(None, scr)
def _down(scr): ModeHandler()._handle_insert_arrow_down(None, scr)
def _page(scr, direction): ModeHandler()._handle_insert_page(None, scr, direction)


def test_up_on_first_line_of_a_multiline_prompt_keeps_it():
    scr = _Scr("one\ntwo", 2, history=["old"])
    _up(scr)
    assert (scr.text, scr._cursor_pos, scr._history_idx) == ("one\ntwo", 2, -1)


def test_down_on_last_line_of_a_new_prompt_keeps_it():
    scr = _Scr("one\ntwo", 7, history=["old"])
    _down(scr)
    assert (scr.text, scr._cursor_pos) == ("one\ntwo", 7)


def test_single_line_draft_survives_a_trip_through_the_history():
    scr = _Scr("draft", 5, history=["new\nest", "older"])
    _up(scr)
    assert scr.text == "new\nest"
    _up(scr)                      # cursor on the last line: moves within the entry
    assert scr.text == "new\nest"
    _up(scr)                      # first line of an untouched recall: older entry
    assert scr.text == "older"
    _down(scr)
    _down(scr)
    assert (scr.text, scr._history_idx) == ("draft", -1)


def test_edited_recall_is_not_replaced():
    scr = _Scr("", 0, history=["a\nb", "older"])
    _up(scr)
    scr._input_buf.append("!")
    scr._cursor_pos = 0
    _up(scr)
    assert scr.text == "a\nb!"
    scr._cursor_pos = len(scr._input_buf)
    _down(scr)
    assert scr.text == "a\nb!"


def test_page_keys_move_a_page_of_lines_and_stop_at_the_ends():
    text = "\n".join(f"l{i}" for i in range(8))
    scr = _Scr(text, len(text), history=["old"])
    _page(scr, -1)
    assert scr.text[:scr._cursor_pos].count("\n") == 4
    _page(scr, -1)
    _page(scr, -1)
    assert scr.text[:scr._cursor_pos].count("\n") == 0
    _page(scr, -1)                # first line of a new multi-line prompt: stays
    assert scr.text == text
    _page(scr, 1)
    assert scr.text[:scr._cursor_pos].count("\n") == 3
