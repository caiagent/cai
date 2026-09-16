"""Tests for the :messages overlay opened while the model is running: the
read-only guard on every mutating key, and the cheap get_revision control op
the live refresh polls."""
from test_wired_agent import FakeApi, make_agent, run_turn, serve

from cai.screen.ansi import KEY_ENTER
from cai.screen.overlays.messages import overlay_nav_key
from cai.screen.state import MsgOverlayCtx


def _ctx(readonly):
    messages = [{"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"}]
    ctx = MsgOverlayCtx(messages)
    ctx.readonly = readonly
    return ctx


def test_readonly_refuses_delete_paste_rewrite_and_edit():
    ctx = _ctx(True)
    ctx.yank_register = [{"role": "user", "content": "pasted"}]
    for key in ("d", "p", "!", KEY_ENTER[0]):
        ctx.status_flash = ""
        assert overlay_nav_key(ctx, key, 40, None) is None
        assert ctx.status_flash == "read-only while the model runs"
    assert len(ctx.messages) == 2
    assert ctx.modified is False
    assert ctx.instruction_mode is False


def test_readonly_still_navigates_and_closes():
    ctx = _ctx(True)
    overlay_nav_key(ctx, "j", 40, None)
    assert ctx.selected_idx == 1
    assert overlay_nav_key(ctx, "q", 40, None) == "close"


def test_editable_delete_still_works():
    ctx = _ctx(False)
    overlay_nav_key(ctx, "d", 40, None)
    assert len(ctx.messages) == 1
    assert ctx.modified is True


def test_control_get_revision_counts_messages(serve):
    wire = serve(make_agent(api=FakeApi(chunks=["ok"])))
    ok, revision, error = wire.control("get_revision")
    assert (ok, revision) == (True, 0)
    run_turn(wire, "hi")
    ok, revision, error = wire.control("get_revision")
    assert (ok, revision) == (True, 2)
