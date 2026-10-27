"""the display renderers: cai.renderer registers per kind on the Environment,
render() chains them (None passes through, a raise is skipped), and the TUI
transcript paints a rendered kind as one block it re-renders in place while
it streams - final once the block closes - while an unrendered kind streams
chunk by chunk through write() exactly as before."""
import pytest

import cai
from cai.environment import Environment, Settings
from cai.events import EventType
from cai.renderers import RenderContext, render
from cai.screen.screen import Screen
from cai.tui import _AttachView, _Transcript, _replay_messages


# --- registration ----------------------------------------------------------

def test_renderer_registers_per_kind_on_the_default_env():
    @cai.renderer("response", "reasoning")
    def r(ctx):
        return ctx.text

    env = Environment.default()
    assert env.renderers("response") == [r]
    assert env.renderers("reasoning") == [r]
    assert env.renderers("user") == []


def test_renderer_rejects_an_unknown_kind_and_no_kind():
    with pytest.raises(ValueError):
        @cai.renderer("assistant")
        def r(ctx):
            pass
    with pytest.raises(ValueError):
        cai.renderer()


def test_renderers_chain_in_registration_order():
    @cai.renderer("response")
    def first(ctx):
        return ctx.text + "1"

    @cai.renderer("response")
    def second(ctx):
        return ctx.text + "2"

    chain = Environment.default().renderers("response")
    assert render(chain, RenderContext(kind="response", text="x")) == "x12"


def test_none_passes_through_and_a_raise_is_skipped(caplog):
    def passthrough(ctx):
        return None

    def broken(ctx):
        raise RuntimeError("boom")

    def upper(ctx):
        return ctx.text.upper()

    out = render([passthrough, broken, upper], RenderContext(kind="response", text="hi"))
    assert out == "HI"
    assert "boom" in caplog.text


def test_non_text_return_is_ignored():
    def wrong(ctx):
        return 42

    assert render([wrong], RenderContext(kind="response", text="hi")) == "hi"


def test_each_renderer_sees_the_context_fields():
    seen = []

    def spy(ctx):
        seen.append((ctx.kind, ctx.text, ctx.done, ctx.width, ctx.style))

    render([spy], RenderContext(kind="response", text="t", done=False, width=42, style="S"))
    assert seen == [("response", "t", False, 42, "S")]


# --- the transcript ----------------------------------------------------------

class _Event:
    def __init__(self, type, text=None, tool_name=None, tool_args=None,
                 tool_result=None, is_error=False):
        self.type = type
        self.text = text
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.is_error = is_error


class _BlockScreen:
    """records the rendered-block surface: write (plain streaming),
    write_block (opens a block, returns its index) and replace_block."""

    def __init__(self):
        self.ops = []
        self._blocks = 0

    def write(self, text, kind=None, block=False):
        self.ops.append(("write", text, kind, block))

    def write_block(self, text, kind=None):
        self.ops.append(("open", text, kind))
        self._blocks += 1
        return self._blocks - 1

    def replace_block(self, index, text, kind=None):
        self.ops.append(("replace", index, text, kind))

    def content_width(self, kind=None):
        return 60


def _env_with(kind, fn):
    env = Environment()
    env.register_renderer(kind, fn)
    return env


def _tag(ctx):
    done = "done" if ctx.done else "live"
    return f"[{done}:{ctx.text}]"


def test_rendered_kind_is_one_block_re_rendered_whole_on_each_chunk():
    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("response", _tag))
    t.event(_Event(EventType.CONTENT, text="hel"))
    t.event(_Event(EventType.CONTENT, text="lo"))
    t.event(_Event(EventType.CONTENT, text=" world"))
    assert screen.ops == [
        ("open", "[live:hel]", Screen.RESPONSE),
        ("replace", 0, "[live:hello]", Screen.RESPONSE),
        ("replace", 0, "[live:hello world]", Screen.RESPONSE),
    ]


def test_block_is_rendered_final_when_the_next_unit_starts():
    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("response", _tag))
    t.event(_Event(EventType.CONTENT, text="answer"))
    t.event(_Event(EventType.TOOL_CALL, tool_name="ls"))
    assert screen.ops[0] == ("open", "[live:answer]", Screen.RESPONSE)
    assert screen.ops[1] == ("replace", 0, "[done:answer]", Screen.RESPONSE)
    # the tool unit has no renderer: it streams through write() as before.
    assert screen.ops[2][0] == "write"
    assert screen.ops[2][2] == Screen.TOOL
    assert screen.ops[2][3] is True


def test_end_of_turn_renders_the_last_block_final():
    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("response", _tag))
    t.event(_Event(EventType.CONTENT, text="bye"))
    t.end()
    assert screen.ops == [("open", "[live:bye]", Screen.RESPONSE),
                          ("replace", 0, "[done:bye]", Screen.RESPONSE)]
    # a second end is a no-op, and the next turn opens a fresh block.
    t.end()
    t.event(_Event(EventType.CONTENT, text="again"))
    assert screen.ops[-1] == ("open", "[live:again]", Screen.RESPONSE)


def test_note_closes_the_open_block_and_renders_itself_final():
    screen = _BlockScreen()
    env = Environment()
    env.register_renderer("response", _tag)
    env.register_renderer("user", _tag)
    t = _Transcript(screen, Settings(), env=env)
    t.event(_Event(EventType.CONTENT, text="a"))
    t.event(_Event(EventType.USER, text="hi"))
    assert screen.ops == [("open", "[live:a]", Screen.RESPONSE),
                          ("replace", 0, "[done:a]", Screen.RESPONSE),
                          ("open", "[done:> hi\n]", Screen.USER)]


def test_unrendered_kind_streams_exactly_as_before():
    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("reasoning", _tag))
    t.event(_Event(EventType.CONTENT, text="hel"))
    t.event(_Event(EventType.CONTENT, text="lo"))
    t.end()
    assert screen.ops == [("write", "hel", Screen.RESPONSE, True),
                          ("write", "lo", Screen.RESPONSE, False)]


def test_leading_newlines_still_stripped_for_a_rendered_block():
    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("response", _tag))
    t.event(_Event(EventType.CONTENT, text="\n\n"))
    t.event(_Event(EventType.CONTENT, text="\nok"))
    assert screen.ops == [("open", "[live:ok]", Screen.RESPONSE)]


def test_a_tool_error_line_keeps_the_open_tool_block():
    # the result line's kind (error) differs from the block's (tool): the
    # block keeps its own kind and style, the text joins it.
    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("tool", _tag))
    t.event(_Event(EventType.TOOL_CALL, tool_name="ls", tool_args={}))
    t.event(_Event(EventType.TOOL_RESULT, tool_name="ls", tool_result="x", is_error=True))
    assert screen.ops[0][0] == "open"
    assert screen.ops[0][2] == Screen.TOOL
    assert screen.ops[1][0] == "replace"
    assert screen.ops[1][3] == Screen.TOOL
    assert "Error" not in screen.ops[1][2]
    assert "<- ls: 1 chars" in screen.ops[1][2]


def test_renderer_width_comes_from_the_screen():
    seen = []

    def spy(ctx):
        seen.append(ctx.width)

    screen = _BlockScreen()
    t = _Transcript(screen, Settings(), env=_env_with("response", spy))
    t.event(_Event(EventType.CONTENT, text="x"))
    assert seen == [60]


def test_transcript_defaults_to_the_process_default_env():
    @cai.renderer("response")
    def tag(ctx):
        return _tag(ctx)

    screen = _BlockScreen()
    t = _Transcript(screen, Settings())
    t.event(_Event(EventType.CONTENT, text="x"))
    assert screen.ops == [("open", "[live:x]", Screen.RESPONSE)]


# --- replay ------------------------------------------------------------------

def test_replay_renders_every_block_final():
    screen = _BlockScreen()
    env = Environment()
    env.register_renderer("response", _tag)
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "two"}]
    _replay_messages(screen, messages, Settings(), env=env)
    rendered = [op for op in screen.ops if op[0] in ("open", "replace")]
    # one render per message, final - never a live pass.
    assert rendered == [("open", "[done:one\n]", Screen.RESPONSE),
                        ("open", "[done:two\n]", Screen.RESPONSE)]
    users = [op for op in screen.ops if op[0] == "write"]
    assert users == [("write", "> hi\n", Screen.USER, True),
                     ("write", "> more\n", Screen.USER, True)]


# --- the real buffer-backed surfaces -----------------------------------------

def test_attach_view_replaces_the_block_in_place():
    view = _AttachView(40)
    view.write("> hi\n", kind=Screen.USER, block=True)
    index = view.write_block("one", Screen.RESPONSE)
    before = view.line_count()
    view.replace_block(index, "one\ntwo\nthree", Screen.RESPONSE)
    lines = [l for l in view.get_lines(0, view.line_count())]
    from cai.screen.ansi import ansi_strip
    plain = [ansi_strip(l) for l in lines]
    assert plain[0].endswith("> hi")
    assert plain[-3:] == ["▌ one", "▌ two", "▌ three"]
    assert view.line_count() == before + 2
    # the next write lands on its own row, after the block.
    view.write("-> ls()\n", kind=Screen.TOOL, block=True)
    plain = [ansi_strip(l) for l in view.get_lines(0, view.line_count())]
    assert plain[-1].endswith("-> ls()")
    assert plain[-2] == ""


def test_attach_view_content_width_excludes_the_gutter():
    view = _AttachView(40)
    assert view.content_width(Screen.RESPONSE) == 38
    assert view.content_width(None) == 40
