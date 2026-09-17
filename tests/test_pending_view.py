"""Tests for the :pending view plumbing: the listable/removable jobs queue,
the row builder, and the open-only-when-something-is-there gate. Offline: no
screen is drawn - the overlay call is stubbed."""
import queue
import threading

import pytest

from cai import tui
from cai.screen import Screen


def test_jobs_put_get_and_empty_timeout():
    jobs = tui._Jobs()
    with pytest.raises(queue.Empty):
        jobs.get(timeout=0.01)
    jobs.put("a")
    jobs.put(tui._CONTINUE)
    assert jobs.get(timeout=0.01) == "a"
    assert jobs.get(timeout=0.01) is tui._CONTINUE
    with pytest.raises(queue.Empty):
        jobs.get(timeout=0.01)


def test_jobs_get_wakes_on_put_from_another_thread():
    jobs = tui._Jobs()
    got = []

    def taker():
        got.append(jobs.get(timeout=2))
    thread = threading.Thread(target=taker)
    thread.start()
    jobs.put("late")
    thread.join(timeout=2)
    assert got == ["late"]


def test_jobs_snapshot_and_guarded_remove():
    jobs = tui._Jobs()
    jobs.put("a")
    jobs.put("b")
    jobs.put("c")
    assert jobs.snapshot() == ["a", "b", "c"]
    assert jobs.remove(1, "b") is True
    assert jobs.snapshot() == ["a", "c"]
    # a stale listing (the worker took the head): index 0 no longer reads "b"
    assert jobs.remove(0, "b") is False
    assert jobs.remove(7, "c") is False
    assert jobs.snapshot() == ["a", "c"]
    assert jobs.get(timeout=0.01) == "a"


def test_pending_nodes_orders_steers_first_and_keeps_ids_stable():
    nodes = tui._pending_nodes(["s1", "s2"], ["q1", tui._CONTINUE, "q1", None])
    kinds = []
    for node in nodes:
        kinds.append((node["kind"], node["index"], node["text"]))
    assert kinds == [("steer", 0, "s1"), ("steer", 1, "s2"),
                     ("queued", 0, "q1"), ("queued", 1, "(continue)"),
                     ("queued", 2, "q1")]
    ids = []
    for node in nodes:
        ids.append(node["id"])
    assert len(set(ids)) == len(ids)          # duplicates of a text get distinct ids
    # removing the first steer renames nothing else
    after = tui._pending_nodes(["s2"], ["q1", tui._CONTINUE, "q1"])
    assert after[0]["id"] == nodes[1]["id"]
    assert after[1]["id"] == nodes[2]["id"]
    assert nodes[3]["item"] is tui._CONTINUE
    for node in nodes:
        assert node["parent"] is None


class _FakeClient:
    def __init__(self, steers):
        self.steers = list(steers)
        self.removed = []

    def get_steer(self):
        return list(self.steers)

    def remove_steer(self, index, text):
        self.removed.append((index, text))
        if index < len(self.steers) and self.steers[index] == text:
            del self.steers[index]
            return True
        return False


class _FakeScreen:
    def __init__(self):
        self.written = []
        self.opened = []

    def write(self, text, kind=None, block=False):
        self.written.append(text)

    def prompt_tree_overlay(self, fetch_fn, **kwargs):
        self.opened.append(kwargs)
        # exercise the callbacks the way the overlay would, then remove one
        nodes = fetch_fn()
        for node in nodes:
            kwargs["label_fn"](node)
            kwargs["color_fn"](node)
            kwargs["preview_fn"](node, 40, 5)
        kwargs["action_fn"](nodes[0]["id"])
        return None


class _FakePending:
    def __init__(self):
        self.removed = 0

    def user_removed(self):
        self.removed += 1


def test_open_pending_writes_a_note_when_nothing_is_pending():
    screen = _FakeScreen()
    jobs = tui._Jobs()
    tui._open_pending(screen, _FakeClient([]), jobs, _FakePending())
    assert screen.opened == []
    assert screen.written == ["[nothing pending]\n"]


def test_open_pending_removes_a_steer_over_the_wire():
    screen = _FakeScreen()
    client = _FakeClient(["steer me"])
    jobs = tui._Jobs()
    jobs.put("later")
    pending = _FakePending()
    tui._open_pending(screen, client, jobs, pending)
    assert len(screen.opened) == 1
    assert screen.opened[0]["title"] == "pending"
    assert client.removed == [(0, "steer me")]
    assert client.steers == []
    assert jobs.snapshot() == ["later"]         # the queued prompt is untouched
    assert pending.removed == 0


def test_open_pending_removes_a_queued_prompt_and_updates_the_chip():
    screen = _FakeScreen()
    jobs = tui._Jobs()
    jobs.put("first")
    jobs.put("second")
    pending = _FakePending()
    tui._open_pending(screen, _FakeClient([]), jobs, pending)
    assert jobs.snapshot() == ["second"]
    assert pending.removed == 1


def test_tab_toggles_status_unless_a_live_element_owns_it():
    from cai.screen.modes import ModeHandler
    from cai.screen.state import CommandException, Mode, TUIState
    from cai.screen.ansi import KEY_TAB

    class _Scr:
        _live = None
    state = TUIState()
    state.mode = Mode.INSERT
    with pytest.raises(CommandException) as exc:
        ModeHandler().handle_key(KEY_TAB, state, _Scr())
    assert str(exc.value) == "status" or exc.value.args[0] == "status"
    state.mode = Mode.NORMAL
    with pytest.raises(CommandException):
        ModeHandler().handle_key(KEY_TAB, state, _Scr())
    # a live ask element (a confirm's yes/no toggles on Tab) keeps the key
    scr = _Scr()
    scr._live = object()
    state.mode = Mode.INSERT
    try:
        ModeHandler().handle_key(KEY_TAB, state, scr)
    except CommandException:
        raise AssertionError("Tab must reach the live element, not toggle status")
    except Exception:
        pass                                          # the stub element has no handle_key


def test_pending_command_and_shortcut_are_registered():
    from cai.screen.modes import _OVERLAY_SHORTCUTS
    from cai.screen.ansi import KEY_CTRL_Q
    assert _OVERLAY_SHORTCUTS[KEY_CTRL_Q] == "pending"
    names = []
    for name, _help in tui._PALETTE_COMMANDS:
        names.append(name)
    assert "pending" in names
    assert "status" in names
