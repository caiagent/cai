"""Tests for the full tool-call rendering (render.render_tool_call): short
values inline on the '->' line, long / multi-line values as labelled blocks
below, python code as its syntax-colored block, and the oversized-value guard."""
from cai.screen.ansi import ansi_strip, SGR_DIM_GRAY
from cai.screen.render import (render_tool_call, INLINE_VALUE_MAX_CHARS,
                               CALL_VALUE_MAX_CHARS)


def _plain(block):
    return ansi_strip(block).splitlines()


def test_short_values_stay_inline_and_untruncated():
    header, block = render_tool_call("fs__read_file", {"file_path": "a.py", "line_start": 1})
    assert header == "-> fs__read_file(file_path=a.py, line_start=1)"
    assert block == ""
    exact = "x" * INLINE_VALUE_MAX_CHARS
    header, block = render_tool_call("t", {"v": exact})
    assert header == f"-> t(v={exact})"
    assert block == ""


def test_long_value_drops_into_a_labelled_block():
    content = "line one\nline two\nline three"
    header, block = render_tool_call("fs__create_file", {"file_path": "n.txt", "content": content})
    assert header == "-> fs__create_file(file_path=n.txt)"
    assert _plain(block) == ["  content:",
                             "    line one",
                             "    line two",
                             "    line three"]
    assert SGR_DIM_GRAY in block
    assert block.endswith("\n")


def test_single_line_over_the_limit_goes_below_too():
    value = "y" * (INLINE_VALUE_MAX_CHARS + 1)
    header, block = render_tool_call("bash", {"command": value})
    assert header == "-> bash()"
    assert _plain(block) == ["  command:", "    " + value]


def test_non_string_values_render_as_json():
    header, block = render_tool_call("t", {"n": 3, "flag": True, "items": ["a", "b"]})
    assert header == "-> t(n=3, flag=true, items=[\"a\", \"b\"])"
    assert block == ""
    big = {}
    for i in range(30):
        big["key" + str(i)] = i
    header, block = render_tool_call("t", {"obj": big})
    assert header == "-> t()"
    lines = _plain(block)
    assert lines[0] == "  obj:"
    assert lines[1] == "    {"
    assert '    "key0": 0,' in lines[2]


def test_python_code_renders_after_other_long_args_without_a_label():
    args = {"code": "print(1)", "timeout": 5}
    header, block = render_tool_call("python", args)
    assert header == "-> python(timeout=5)"
    assert _plain(block) == ["  print(1)"]
    args = {"code": "print(2)", "note": "a\nb"}
    header, block = render_tool_call("python", args)
    assert _plain(block) == ["  note:", "    a", "    b", "  print(2)"]


def test_oversized_value_is_cut_with_a_pointer_to_messages():
    blob = "z" * (CALL_VALUE_MAX_CHARS + 500)
    header, block = render_tool_call("mcp__upload", {"data": blob})
    lines = _plain(block)
    assert lines[0] == "  data:"
    assert len(lines[1]) == 4 + CALL_VALUE_MAX_CHARS
    assert lines[2] == "    … +500 chars, see :messages"


def test_no_args():
    assert render_tool_call("t", {}) == ("-> t()", "")
    assert render_tool_call("t", None) == ("-> t()", "")
