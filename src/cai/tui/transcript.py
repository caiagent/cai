"""tui.transcript: painting a conversation into a Screen - the live event
stream (_Transcript), a stored conversation replayed the same way, and the
read-only view an attach overlay paints into (_AttachView)."""
import json

from cai.origin import origin_label
from cai.origin import origin_of
from cai.environment import Environment
from cai.environment import Settings
from cai.events import EventType
from cai.renderers import RenderContext
from cai.renderers import render
from cai.screen import Screen
from cai.screen.ansi import SGR_RESET
from cai.screen.buffer import ContentBuffer
from cai.screen.render import render_tool_call


class _Transcript:
    """renders agent events into the conversation viewport with uniform
    spacing: exactly one blank line between logical blocks.

    a block is a user prompt, an assistant message, a reasoning trace, a tool
    exchange, or a one-off note. consecutive CONTENT (or REASONING) chunks
    stream into one block, and a tool call chains with its result (and any
    directly following tool exchange) into one tool block - only a change of
    unit starts a new block (block=True). a block's opening chunk is stripped
    of leading newlines (the boundary is owned by the screen, which also
    trims whatever trailing newlines the previous block streamed), so the
    model's own paragraph breaks survive while the gaps between blocks never
    depend on the newlines each event happened to carry.

    a kind with renderers registered (cai.renderer - the env's, the TUI's
    process-wide default unless one is given) is painted through them: the
    transcript keeps the open block's raw text, and on every chunk re-renders
    the WHOLE block (RenderContext.done False) and replaces it on screen in
    place - one buffer segment, opened with write_block and swapped with
    replace_block - then renders it once more, final (done True), when the
    block closes: the next unit starting, a note, or close() at the end of a
    turn. a kind with no renderer streams straight through write(), chunk by
    chunk, exactly as before."""

    def __init__(self, screen, settings, env=None):
        self._screen = screen
        self._settings = settings   # the live env Settings (:config edits it)
        if env is None:
            env = Environment.default()
        self._env = env
        self._streaming = None   # the open unit: "content"/"reasoning"/"tool"/None
        # the open rendered block, when its kind has renderers: its raw text
        # so far, its kind, and the screen segment it occupies.
        self._raw = ""
        self._kind = None
        self._seg = None

    def event(self, event):
        if event.type == EventType.CONTENT:
            self._stream(event.text or "", "content", Screen.RESPONSE)
            return
        if event.type == EventType.REASONING:
            if not self._settings.show_reasoning:
                return
            self._stream(event.text or "", "reasoning", Screen.REASONING)
            return
        if event.type == EventType.USER:
            self._user_turn((event.text or "").rstrip(), getattr(event, "origin", None))
            return
        if event.type == EventType.TOOL_CALL:
            header, block = render_tool_call(event.tool_name, event.tool_args)
            self._stream(header + "\n", "tool", Screen.TOOL)
            if block:
                self._stream(block, "tool", Screen.TOOL)
            return
        if event.type == EventType.TOOL_RESULT:
            kind = Screen.TOOL
            if event.is_error:
                kind = Screen.ERROR
            result = event.tool_result or ""
            self._stream(f"<- {event.tool_name}: {len(result)} chars\n", "tool", kind)
            return
        if event.type == EventType.RETRY:
            self.note(f"[{event.text}]\n", Screen.META)
            return

    def _stream(self, text, unit, kind, final=False):
        # a fresh unit starts a new block; the same unit continues the open
        # one. the opening chunk sheds its leading newlines - a chunk that is
        # nothing but newlines does not open the block at all, the next real
        # chunk does. final=True says the text completes its block (a replayed
        # message, a note): a rendered block is then painted done at once.
        block = unit != self._streaming
        if block:
            text = text.lstrip('\n')
            if not text:
                return
            self.close()
        self._streaming = unit
        chain = self._env.renderers(kind)
        if not chain and self._seg is None:
            self._screen.write(text, kind=kind, block=block)
            return
        # a block painted through renderers: accumulate, re-render whole,
        # replace in place. a continuation whose kind differs from the open
        # block's (a tool error line inside a tool block) keeps the block's
        # kind - the block is one segment with one style.
        if self._seg is None:
            self._raw = text
            self._kind = kind
            self._seg = self._screen.write_block(self._render(final), kind)
        else:
            self._raw += text
            self._screen.replace_block(self._seg, self._render(final), self._kind)
        if final:
            self._forget()

    def _render(self, done):
        ctx = RenderContext(kind=self._kind,
                            text=self._raw,
                            done=done,
                            width=self._screen.content_width(self._kind),
                            style=Screen._KIND_STYLES.get(self._kind, ''))
        return render(self._env.renderers(self._kind), ctx)

    def close(self):
        """end the open block: a rendered one is painted final (done=True).
        the worker calls it when a turn's RESULT lands, so the last block of
        a turn gets its final render even though no next block follows."""
        if self._seg is None:
            return
        self._screen.replace_block(self._seg, self._render(True), self._kind)
        self._forget()

    def _forget(self):
        self._seg = None
        self._raw = ""
        self._kind = None

    def end(self):
        """a turn ended: close the open block and forget the unit, so the
        next turn's first chunk opens a fresh block."""
        self.close()
        self._streaming = None

    def _user_turn(self, text, origin):
        """paint a user-role turn: the user's own as a '> ' prompt; one the
        harness injected (a sub-agent's report - see cai.origin) as a meta
        block headed by who wrote it, so it never reads as something typed."""
        if origin is None:
            self.note(f"> {text}\n", Screen.USER)
            return
        self.note(f"[report from {origin_label(origin)}]\n{text}\n", Screen.META)

    def note(self, text, kind):
        """write a self-contained block (a prompt, a tool line, a notice),
        always separated from what precedes it by one blank line."""
        self._stream(text, "note", kind, final=True)
        self._streaming = None

    def replay(self, messages):
        """paint a stored conversation the way the live stream paints it:
        user turns, the assistant's stored reasoning (dimmed, when the
        settings say to show it) then its content, its tool calls as '->'
        lines, each tool reply chaining on as '<- name: N chars' (the name
        mapped back through the call id, which is all a tool message
        carries). every block is final, so a rendered kind gets its done
        render at once."""
        call_names = {}
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)
            if role == "user":
                self._user_turn(content, origin_of(message))
                continue
            if role == "tool":
                name = call_names.get(message.get("tool_call_id"), "?")
                self._stream(f"<- {name}: {len(content)} chars\n", "tool", Screen.TOOL)
                continue
            if role != "assistant":
                continue
            reasoning = message.get("_reasoning")
            if reasoning and self._settings.show_reasoning:
                self._stream(reasoning + "\n", "reasoning", Screen.REASONING, final=True)
            if content.strip():
                self._stream(content + "\n", "content", Screen.RESPONSE, final=True)
            for call in (message.get("tool_calls") or []):
                function = call.get("function") or {}
                name = function.get("name") or "?"
                call_names[call.get("id")] = name
                args = _stored_call_args(function)
                header, body = render_tool_call(name, args)
                self._stream(header + "\n", "tool", Screen.TOOL)
                if body:
                    self._stream(body, "tool", Screen.TOOL)
        self.end()


class _AttachView:
    """the write() surface the attach overlay paints from: the same
    write(text, kind=, block=) contract Screen offers - so _replay_messages
    and _Transcript render a watched agent's conversation into it unchanged -
    backed by a private ContentBuffer instead of the terminal. the overlay
    reads the wrapped lines back (get_lines/line_count/rewrap) and paints
    them itself; `version` bumps on every write so it knows when to repaint.
    the kind styling (SGR transitions, the colored gutter) delegates to the
    Screen methods that own it, so both views render identically."""

    _KIND_STYLES = Screen._KIND_STYLES
    _KIND_GUTTERS = Screen._KIND_GUTTERS

    def __init__(self, cols):
        self._cols = cols
        self._buffer = ContentBuffer(cols)
        self._current_kind = None
        self.version = 0

    def write(self, text, kind=None, block=False):
        if not text: return
        if block:
            self._start_block()
        if kind is not None:
            text = self._apply_kind(text, kind, separated=block)
        self._buffer.append_text(text, gutter=self._kind_gutter(kind))
        self.version += 1

    # the rendered-block surface (write_block / replace_block /
    # content_width), the same contract Screen offers, so a transcript
    # painting through renderers lands here unchanged.
    def write_block(self, text, kind=None):
        self._start_block()
        prev = self._current_kind
        prefix = ''
        if kind != prev and prev is not None and self._KIND_STYLES.get(prev):
            prefix = SGR_RESET
        self._current_kind = kind
        self._buffer.append_text(prefix + Screen._block_text(self, text, kind),
                                 gutter=self._kind_gutter(kind))
        self.version += 1
        return self._buffer.segment_count() - 1

    def replace_block(self, index, text, kind=None):
        if index < 0 or index >= self._buffer.segment_count():
            return
        self._buffer.replace_segment(index, Screen._block_text(self, text, kind))
        self.version += 1

    def content_width(self, kind=None):
        return Screen.content_width(self, kind)

    def _apply_kind(self, text, kind, separated=False):
        return Screen._apply_kind(self, text, kind, separated=separated)

    def _kind_gutter(self, kind):
        return Screen._kind_gutter(self, kind)

    def _start_block(self):
        # Screen._start_block minus its cursor/viewport clamping - the overlay
        # owns the viewport here.
        self._buffer.end_line()
        self._buffer.trim_trailing_blanks()
        if self._buffer.line_count() > 0:
            self._buffer.append_text('\n')

    def rewrap(self, cols):
        self._cols = cols
        self._buffer.rewrap(cols)

    def line_count(self):
        return self._buffer.line_count()

    def get_lines(self, start, count):
        return self._buffer.get_lines(start, count)


def _stored_call_args(function):
    """the parsed args of a stored tool call, for a replayed '->' line; a blob
    that doesn't parse renders as empty args rather than raising."""
    try:
        args = json.loads(function.get("arguments") or "{}")
    except ValueError:
        return {}
    if not isinstance(args, dict):
        return {}
    return args


def _replay_messages(screen, messages, cfg=None, env=None):
    """write a conversation into the (cleared) viewport so the user sees what
    they switched to - the painting :redraw, :load, and a :history fork share.
    turns render like the live transcript (_Transcript.replay), renderers
    included; cfg is the live env Settings (show_reasoning - hidden when no
    cfg is given), env the Environment whose renderers paint it (the process
    default when None)."""
    if cfg is None:
        cfg = Settings()
        cfg.show_reasoning = False
    _Transcript(screen, cfg, env=env).replay(messages)
