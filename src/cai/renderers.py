"""renderers: the display-side extension point - how an extension reshapes a
block of conversation text before the TUI paints it.

A renderer is a pure function from text to text, registered per kind with the
cai.renderer decorator (onto the current Environment, like a hook or a
command):

    @cai.renderer("response")
    def markdown(ctx: cai.RenderContext):
        return md_to_ansi(ctx.text, width=ctx.width, base=ctx.style)

The TUI transcript calls the chain for a kind with the WHOLE block's text so
far - never a chunk - every time the block grows, and once more when it
closes, replacing the block on screen in place each time. so a renderer sees
the full markdown of an assistant message as it streams (ctx.done False: a
fence may still be open, a '**' unmatched) and then the final text (ctx.done
True). it returns the rendered text (ANSI allowed) or None to leave the text
as it is. renderers for one kind chain in registration order, each handed the
previous one's output, so the user's init.py - loaded last - has the final
word. a renderer that raises is logged and skipped, so a broken one never
takes the transcript down.

this is a client-side concern: the TUI is a wire client and may be attached
to an agent in another process, so renderers are NOT hooks (those fire in the
agent's loop) - they live on the Environment the TUI loads at startup and
never touch the conversation itself; search, yank and the :messages overlay
see what each surface shows. a kind is one of the screen's text kinds
(RENDER_KINDS): user, response, reasoning, tool, meta, error."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from cai.environment import Environment


log = logging.getLogger("cai")

# the text kinds a renderer may claim - the Screen kind constants, spelled out
# here so an extension never imports the screen package to name one.
RENDER_KINDS = ("user", "response", "reasoning", "tool", "meta", "error")


@dataclass(frozen=True)
class RenderContext:
    """what a renderer receives: the block's text so far (ctx.text), whether
    it is final (ctx.done) or still streaming, the columns the block may use
    (ctx.width - a hint: the screen rewraps long lines, and a resize does not
    re-render), the block's base SGR style (ctx.style - re-open it after an
    SGR reset so the surrounding text keeps its color), and the kind."""
    kind: str
    text: str
    done: bool = True
    width: int = 80
    style: str = ""


def renderer(*kinds):
    """decorator: register a renderer for one or more text kinds on the current
    Environment, e.g. @cai.renderer("response") or @cai.renderer("response", "reasoning").
    the function is called with a RenderContext and returns the rendered text
    or None. it lands on the env being load()ed - else the process default -
    so once Environment.load() imports the extensions the TUI's transcript
    renders through it. an unknown kind raises ValueError."""
    if not kinds:
        raise ValueError(f"cai.renderer needs at least one kind: {RENDER_KINDS}")
    for kind in kinds:
        if kind in RENDER_KINDS: continue
        raise ValueError(f"unknown render kind: {kind!r}. valid: {RENDER_KINDS}")

    def decorator(fn):
        for kind in kinds:
            Environment.target().register_renderer(kind, fn)
        return fn
    return decorator


def render(chain, ctx):
    """run `chain` (the renderers of ctx.kind, in order) over ctx.text and
    return the result: each renderer's non-None output becomes the next one's
    ctx.text; None leaves the text untouched; a raise is logged and skipped.
    with an empty chain this is the identity."""
    text = ctx.text
    for fn in chain:
        step = RenderContext(kind=ctx.kind,
                             text=text,
                             done=ctx.done,
                             width=ctx.width,
                             style=ctx.style)
        name = getattr(fn, "__name__", repr(fn))
        try:
            out = fn(step)
        except Exception:
            log.exception("renderer %r for %s raised", name, ctx.kind)
            continue
        if out is None: continue
        if not isinstance(out, str):
            log.warning("renderer %r for %s returned %r, not text - ignored",
                        name, ctx.kind, type(out).__name__)
            continue
        text = out
    return text
