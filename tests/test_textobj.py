"""the vim text objects behind `vi(` / `vi"` in the content buffer: the bracket
branch must refuse a column with no character under it instead of indexing
past the line."""
from cai.screen.modes import _textobj_inner_delimited


def test_bracket_on_empty_line_is_none():
    assert _textobj_inner_delimited("", 0, "(") is None


def test_bracket_with_cursor_past_end_is_none():
    assert _textobj_inner_delimited("abc", 3, "(") is None


def test_bracket_inner_range():
    assert _textobj_inner_delimited("(abc)", 0, "(") == (1, 3)
    assert _textobj_inner_delimited("(abc)", 4, ")") == (1, 3)
    assert _textobj_inner_delimited("x (a (b) c) y", 5, "(") == (6, 6)


def test_quote_inner_range():
    assert _textobj_inner_delimited('say "hi"', 5, '"') == (5, 6)
    assert _textobj_inner_delimited('""', 0, '"') is None
