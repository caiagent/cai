"""Tests for screen.togglelist.ToggleList - the checkbox list state behind the
tools/skills overlay and the inline pick-many prompt."""
from cai.screen.ansi import KEY_ESC, KEY_ENTER, KEY_CTRL_D, KEY_CTRL_U, ansi_strip
from cai.screen.togglelist import ToggleList


def _list(names=("alpha", "beta", "gamma"), checked=("beta",)):
    entries = [(name, "ext") for name in names]
    return ToggleList(entries, checked)


def test_navigation_and_toggle():
    lst = _list()
    assert lst.selected_idx == 0
    assert lst.handle_key("j") is False
    assert lst.selected_idx == 1
    lst.handle_key(" ")
    assert "beta" not in lst.checked
    lst.handle_key(" ")
    assert "beta" in lst.checked
    lst.handle_key("G")
    assert lst.selected_idx == 2
    lst.handle_key("g")
    lst.handle_key("g")
    assert lst.selected_idx == 0
    lst.handle_key(KEY_CTRL_D, page=2)
    assert lst.selected_idx == 2
    lst.handle_key(KEY_CTRL_U, page=2)
    assert lst.selected_idx == 0


def test_close_keys():
    assert _list().handle_key(KEY_ENTER[0]) is True
    assert _list().handle_key(KEY_ESC) is True


def test_search_moves_cursor_and_enter_keeps_it():
    lst = _list()
    lst.handle_key("/")
    assert lst.search_mode is True
    lst.handle_key("g")
    lst.handle_key("a")
    assert lst.search_matches == [2]
    assert lst.selected_idx == 2
    assert lst.handle_key(KEY_ENTER[0]) is False
    assert lst.search_mode is False
    assert lst.selected_idx == 2
    lst.handle_key("/")
    lst.handle_key("x")
    lst.handle_key(KEY_ESC)
    assert lst.selected_idx == 2
    assert lst.search_pattern == ""


def test_rows_pad_scroll_and_drop_empty_tag():
    lst = _list()
    rows = lst.rows(30, 4)
    assert len(rows) == 4
    assert ansi_strip(rows[0]).startswith("  [ ] alpha")
    assert "[x] beta" in ansi_strip(rows[1])
    assert "[ext]" in ansi_strip(rows[1])
    assert ansi_strip(rows[3]).strip() == ""
    lst.selected_idx = 2
    assert "gamma" in ansi_strip(lst.rows(30, 1)[0])
    bare = ToggleList([("one", "")], [])
    assert "[" not in ansi_strip(bare.rows(20, 1)[0])[6:]


def test_status_and_checked_helpers():
    lst = _list()
    assert lst.status(80).startswith(" 1/3 enabled")
    lst.handle_key("?")
    lst.handle_key("z")
    assert lst.status(80) == " ?z [no match]"
    lst.handle_key(KEY_ESC)
    lst.checked = {"gamma", "alpha", "zzz"}
    assert lst.checked_in_order() == ["alpha", "gamma"]
    assert lst.checked_count() == 2
