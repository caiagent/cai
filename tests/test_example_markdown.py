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


def _load():
    shutil.copytree(os.path.join(EXAMPLES, "markdown"), os.path.join(extensions_dir(), "markdown"))
    env = Environment().load()
    chain = env.renderers("response")
    assert [fn.__name__ for fn in chain] == ["markdown"]
    return chain[0]


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
