"""the markdown example under examples/extensions loads as a bundle and
renders, fully offline: copied into a temp HOME's extensions dir,
Environment.load() imports it, and its renderer is called with hand-built
contexts - finished text and the partial text a streaming block hands it."""
import os
import shutil

import pytest

from cai.environment import Environment, extensions_dir
from cai.renderers import RenderContext
from cai.screen.ansi import ansi_strip


EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples", "extensions")

_BOLD = "\033[1m"
_RESET = "\033[m"
_BASE = "\033[36m"


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


# the renderer has two passes: 'parsed' (markdown-it-py, when installed) and
# 'plain' (stdlib regexes, the fallback). the shared tests below run against
# both, so the fallback keeps rendering what the parser does.
_MODE = {"pass": "plain"}


@pytest.fixture(autouse=True, params=["plain", "parsed"])
def _pass(request):
    if request.param == "parsed":
        pytest.importorskip("markdown_it")
    _MODE["pass"] = request.param


def _load(mode=None):
    """load the bundle and return its renderer, pinned to one pass: the
    `mode` given, else the one the current test is parametrized with."""
    shutil.copytree(os.path.join(EXAMPLES, "markdown"), os.path.join(extensions_dir(), "markdown"))
    env = Environment().load()
    chain = env.renderers("response")
    assert [fn.__name__ for fn in chain] == ["markdown"]
    fn = chain[0]
    if (mode or _MODE["pass"]) == "plain":
        fn.__globals__["_PARSER"] = None
    else:
        assert fn.__globals__["_PARSER"] is not None
    return fn


def _render(fn, text, done=True, width=60):
    return fn(RenderContext(kind="response", text=text, done=done, width=width, style=_BASE))


def _plain(text):
    return ansi_strip(text)


def test_bundle_registers_a_renderer_for_response_only():
    shutil.copytree(os.path.join(EXAMPLES, "markdown"), os.path.join(extensions_dir(), "markdown"))
    env = Environment().load()
    assert [fn.__name__ for fn in env.renderers("response")] == ["markdown"]
    assert env.renderers("reasoning") == []
    assert env.renderers("user") == []


def test_headings_are_bold_with_the_hashes_dropped():
    md = _load()
    out = _render(md, "## Plan\nbody")
    assert _plain(out) == "Plan\nbody"
    assert out.startswith(_BOLD + "Plan" + _RESET + _BASE)


def test_bullets_and_tasks():
    md = _load()
    out = _render(md, "- one\n  * two\n- [ ] todo\n- [x] done\n1. first\n2) second")
    assert _plain(out) == "• one\n  • two\n☐ todo\n☑ done\n1. first\n2) second"


def test_inline_emphasis_reopens_the_base_style():
    md = _load()
    out = _render(md, "a **bold** word")
    assert _plain(out) == "a bold word"
    assert out == "a " + _BOLD + "bold" + _RESET + _BASE + " word"


def test_inline_code_is_styled_and_its_contents_left_alone():
    md = _load()
    out = _render(md, "use `**not bold**` here")
    assert _plain(out) == "use **not bold** here"
    assert _BOLD not in out


def test_links_show_text_and_url():
    md = _load()
    out = _render(md, "see [docs](https://x.y/z)")
    assert _plain(out) == "see docs (https://x.y/z)"


def test_fenced_code_is_framed_and_labelled():
    md = _load()
    out = _render(md, "```python\nprint(1)\n```\nafter")
    assert _plain(out) == "┌─ python\n│ print(1)\n└─\nafter"


def test_code_block_contents_are_not_markdown():
    md = _load()
    out = _render(md, "```\n# not a heading\n- not a bullet\n```")
    assert _plain(out) == "┌─\n│ # not a heading\n│ - not a bullet\n└─"
    assert _BOLD not in out


def test_an_open_fence_renders_while_streaming():
    md = _load()
    out = _render(md, "```sh\nls -la", done=False)
    assert _plain(out) == "┌─ sh\n│ ls -la"


def test_unmatched_markers_stay_literal_while_streaming():
    md = _load()
    assert _plain(_render(md, "this is **bo", done=False)) == "this is **bo"
    assert _plain(_render(md, "a `tick", done=False)) == "a `tick"


def test_tables_align_columns_and_rule_the_header():
    md = _load()
    text = "| name | n |\n|---|---|\n| alpha | 1 |\n| b | 22 |"
    out = _plain(_render(md, text))
    assert out.split("\n") == ["name  │ n",
                               "──────┼───",
                               "alpha │ 1",
                               "b     │ 22"]


def test_a_partial_table_renders_the_rows_so_far():
    md = _load()
    out = _plain(_render(md, "| a | b |\n|---|---|\n| 1 |", done=False))
    assert out.split("\n")[0] == "a │ b"
    assert out.split("\n")[-1].startswith("1")


def test_quotes_and_rules():
    md = _load()
    out = _plain(_render(md, "> quoted\n---\nplain", width=20))
    assert out == "▎ quoted\n────────────────────\nplain"


def test_blank_lines_and_trailing_newline_survive():
    md = _load()
    assert _plain(_render(md, "one\n\ntwo\n")) == "one\n\ntwo\n"


def test_plain_text_is_unchanged():
    md = _load()
    text = "just a sentence, with 2 * 3 = 6 and a_b_c in it."
    assert _render(md, text) == text


# --- the parsed pass only: what a real parser gets right ----------------------

def _parsed():
    pytest.importorskip("markdown_it")
    return _load("parsed")


def test_parsed_nested_emphasis_composes():
    md = _parsed()
    out = _render(md, "*it **both** it*")
    assert _plain(out) == "it both it"
    italic = "\033[3m"
    assert italic + _BOLD + "both" + _RESET + _BASE in out


def test_parsed_stray_markers_do_not_style_the_line():
    md = _parsed()
    text = "glob *.py and *.md, then snake_case_name and 2 * 3"
    assert _render(md, text) == text


def test_parsed_table_columns_are_measured_on_the_rendered_cells():
    md = _parsed()
    text = "| key | value |\n|---|---|\n| `a` | **bold** |\n| long-key | x |"
    lines = _plain(_render(md, text)).split("\n")
    assert lines == ["key      │ value",
                     "─────────┼──────",
                     "a        │ bold",
                     "long-key │ x"]


def test_parsed_table_honours_column_alignment():
    md = _parsed()
    text = "| n | total |\n|:--|--:|\n| a | 5 |\n| b | 1200 |"
    lines = _plain(_render(md, text)).split("\n")
    assert lines[2] == "a │     5"
    assert lines[3] == "b │  1200"


def test_parsed_wide_characters_keep_a_table_aligned():
    md = _parsed()
    text = "| a | b |\n|---|---|\n| 日本 | x |\n| ab | y |"
    lines = _plain(_render(md, text)).split("\n")
    assert lines[2] == "日本 │ x"
    assert lines[3] == "ab   │ y"


def test_parsed_ordered_list_numbers_and_hanging_indent():
    md = _parsed()
    text = "9. nine\n10. ten\n    more of ten"
    assert _plain(_render(md, text)) == " 9. nine\n10. ten\n    more of ten"


def test_parsed_code_block_inside_a_list_item_is_indented_with_it():
    md = _parsed()
    text = "- step\n\n  ```sh\n  ls\n  ```\n- next"
    assert _plain(_render(md, text)) == "• step\n\n  ┌─ sh\n  │ ls\n  └─\n• next"


def test_parsed_quote_holds_nested_blocks():
    md = _parsed()
    assert _plain(_render(md, "> note\n> - item")) == "▎ note\n▎ • item"


def test_parsed_rule_under_a_line_is_a_rule_not_a_heading():
    md = _parsed()
    out = _render(md, "plain\n---", width=10)
    assert _plain(out) == "plain\n──────────"
    assert _BOLD not in out


def test_parsed_autolink_shows_the_url_once():
    md = _parsed()
    assert _plain(_render(md, "see <https://x.y/z>")) == "see https://x.y/z"


# --- code highlighting (pygments) -----------------------------------------------

_KEYWORD = "\033[38;5;176m"
_STRING = "\033[38;5;114m"
_ONE_COLOR = "\033[38;5;215m"


def test_fenced_code_is_highlighted_by_its_language():
    pytest.importorskip("pygments")
    md = _load()
    out = _render(md, '```python\ndef f():\n    return "s"\n```')
    assert _plain(out) == '┌─ python\n│ def f():\n│     return "s"\n└─'
    assert _KEYWORD + "def" + _RESET in out
    assert _KEYWORD + "return" + _RESET in out
    assert _STRING + '"s"' + _RESET in out


def test_a_multiline_token_is_colored_on_every_line():
    pytest.importorskip("pygments")
    md = _load()
    out = _render(md, '```python\nx = """a\nb"""\n```')
    lines = out.split("\n")
    assert _STRING + '"""a' + _RESET in lines[1]
    assert _STRING + 'b"""' + _RESET in lines[2]


def test_an_unknown_language_keeps_the_single_code_color():
    md = _load()
    out = _render(md, "```nosuchlang\nx = 1\n```")
    assert _ONE_COLOR + "x = 1" + _RESET in out


def test_code_keeps_one_color_without_pygments():
    md = _load()
    md.__globals__["get_lexer_by_name"] = None
    md.__globals__["_lexer"].cache_clear()
    md.__globals__["_code_lines"].cache_clear()
    out = _render(md, "```python\ndef f(): pass\n```")
    assert _ONE_COLOR + "def f(): pass" + _RESET in out
    assert _KEYWORD not in out


def test_code_lines_keep_their_blank_lines_and_indentation():
    md = _load()
    out = _render(md, "```python\n\nif x:\n\n    y = 1\n\n```")
    assert _plain(out) == "┌─ python\n│ \n│ if x:\n│ \n│     y = 1\n│ \n└─"
