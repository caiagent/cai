"""Tests for cai.current_ui() - the one accessor an in-process tool, hook or
command uses to reach the human - and the multiselect primitive."""
from cai.hooks import HookContext, HookEvent, HooksRegistry
from cai.llm import _handle_tool_calls
from cai.ui import BaseUI, NULL_UI, TerminalUI, current_ui, reset_ui, set_ui
from cai.wire import Wire
from cai.wired_agent import WireUI


class RecordingUI(BaseUI):
    interactive = True

    def __init__(self):
        self.calls = []

    def multiselect(self, message, options, *, default=(), detail=""):
        self.calls.append((message, list(options), list(default)))
        return ["b"]


def test_current_ui_is_null_outside_a_run():
    assert current_ui() is NULL_UI


def test_set_and_reset_ui():
    ui = RecordingUI()
    token = set_ui(ui)
    assert current_ui() is ui
    reset_ui(token)
    assert current_ui() is NULL_UI
    token = set_ui(None)
    assert current_ui() is NULL_UI
    reset_ui(token)


def test_hooks_fire_publishes_ctx_ui():
    ui = RecordingUI()
    seen = []
    registry = HooksRegistry()
    registry.register("after_turn", lambda ctx: seen.append(current_ui()))
    ctx = HookContext(event=HookEvent.AFTER_TURN, messages=[], model="m", ui=ui)
    registry.fire(HookEvent.AFTER_TURN, ctx)
    assert seen == [ui]
    assert current_ui() is NULL_UI


def test_tool_dispatch_publishes_run_ui():
    ui = RecordingUI()
    seen = []

    def dispatch(name, args):
        seen.append(current_ui())
        return "ok"

    call = {}
    call["id"] = "1"
    call["name"] = "t"
    call["arguments"] = "{}"
    call["args"] = {}
    call["valid"] = True
    events = list(_handle_tool_calls([call], [], "", None, dispatch, HooksRegistry(),
                                     "m", None, ui, None, None))
    assert seen == [ui]
    assert events[-1].tool_result == "ok"
    assert current_ui() is NULL_UI


def test_base_multiselect_returns_defaults_in_option_order():
    assert BaseUI().multiselect("q", ["a", "b", "c"], default=["c", "a", "zzz"]) == ["a", "c"]
    assert BaseUI().multiselect("q", ["a", "b"]) == []


def test_terminal_multiselect_parses_comma_numbers(monkeypatch):
    ui = TerminalUI()
    ui.interactive = True
    monkeypatch.setattr("builtins.input", lambda: "3, 1, x, 9, 1")
    assert ui.multiselect("q", ["a", "b", "c"]) == ["c", "a"]
    monkeypatch.setattr("builtins.input", lambda: "")
    assert ui.multiselect("q", ["a", "b", "c"], default=["b"]) == ["b"]


def test_wire_answer_multiselect_replies_with_list():
    class Channel:
        def __init__(self):
            self.sent = b""

        def sendall(self, data):
            self.sent += data

    ch = Channel()
    ui = RecordingUI()
    msg = {}
    msg["type"] = Wire.PROMPT
    msg["id"] = "7"
    msg["kind"] = "multiselect"
    msg["message"] = "which?"
    msg["options"] = ["a", "b"]
    msg["default"] = ["a"]
    assert Wire(ch).answer(msg, ui) is True
    assert ui.calls == [("which?", ["a", "b"], ["a"])]
    assert b'"value": ["b"]' in ch.sent


def test_wireui_multiselect_keeps_only_known_options():
    class Sender:
        def send_prompt(self, *args, **kwargs):
            pass

    ui = WireUI(Sender())
    ui._prompt = lambda *a, **k: (True, ["zzz", "b"])
    assert ui.multiselect("q", ["a", "b"]) == ["b"]
    ui._prompt = lambda *a, **k: (False, None)
    assert ui.multiselect("q", ["a", "b"], default=["a"]) == ["a"]
