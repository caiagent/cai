"""tui.client: the wire side of a tab - AgentClient (the streaming and control
connections to a served agent), ScreenUI (the agent's questions answered on the
screen), the jobs queue and the worker thread that submits prompts and renders
the event stream."""
import logging
import queue
import select
import threading

from cai import usage
from cai.channel import connect
from cai.events import EventType
from cai.screen import Screen
from cai.ui import BaseUI
from cai.wire import Wire
from cai.tui.transcript import _Transcript


log = logging.getLogger("cai")


# jobs-queue sentinel: a continue turn (re-enter the agentic loop with no new user
# message), enqueued after a :history fork that lands on an unfinished point.
_CONTINUE = object()


# how long the worker waits on the stream wire before checking the jobs queue
# for a local prompt to submit.
_PUMP_TIMEOUT = 0.1


class AgentClient:
    """drives a served agent purely over the wire. a streaming connection carries
    runs (SUBMIT and the EVENT/RESULT stream the worker drains); a separate,
    lock-guarded control connection carries the request/response state ops. the
    TUI talks to the agent only through this, so the very same surface drives a
    local served agent today and a remote one over --attach later."""

    def __init__(self, path):
        self.stream = Wire(connect(path))
        self._ctrl = Wire(connect(path))
        self._ctrl_lock = threading.Lock()
        # control ops the agent refused, waiting for drain_errors to hand
        # them to the screen alongside the agent's own.
        self._failed = []

    def submit(self, text):
        self.stream.send_submit(text)

    def continue_run(self):
        """re-enter the agentic loop with no new user turn. a None-text SUBMIT
        runs agent.run(None) on the host - used after a :history fork lands on an
        unfinished point (a user turn, a tool result, or a pending tool call)."""
        self.stream.send_submit(None)

    def steer(self, text, origin=None):
        """steer `text` into the agent: a running turn folds it in at its
        next step, an idle agent runs it as a turn. `origin` (cai.origin)
        marks text the user did not type here."""
        self.stream.send_steer(text, origin=origin)

    def interrupt(self):
        self.stream.send_interrupt()

    def recv(self):
        return self.stream.recv()

    def _call(self, op, value=None):
        """one request/response control op, serialized so the worker's token
        polls never interleave with a main-thread overlay op on the one
        control connection. returns the op's value, or None on failure."""
        with self._ctrl_lock:
            ok, result, error = self._ctrl.control(op, value)
        if not ok:
            self._control_failed(op, error)
            return None
        return result

    def _control_failed(self, op, error):
        """log a refused control op and queue it for the screen. a dropped
        connection is not queued: every poll would repeat it, and the TUI is
        on its way out anyway."""
        log.warning("control %r failed: %s", op, error)
        if error == "disconnected": return
        message = f"{op} failed: {error}"
        with self._ctrl_lock:
            if message in self._failed: return
            self._failed.append(message)

    def get_info(self):
        return self._call("get_info") or {}

    def get_steer(self):
        """the steered texts queued on the agent, in delivery order."""
        return self._call("get_steer") or []

    def remove_steer(self, index, text):
        """drop the queued steer at index (guarded by its text); True if it went."""
        value = {}
        value["index"] = index
        value["text"] = text
        return self._call("remove_steer", value) is True

    def get_messages(self):
        return self._call("get_messages") or []

    def get_revision(self):
        """the conversation's change counter (its length) - cheap to poll."""
        return self._call("get_revision") or 0

    def set_messages(self, messages):
        self._call("set_messages", messages)

    def get_selected_tools(self):
        return self._call("get_selected_tools") or []

    def get_available_tools(self):
        return self._call("get_available_tools") or []

    def set_selected_tools(self, names):
        self._call("set_selected_tools", names)

    def get_selected_skills(self):
        return self._call("get_selected_skills") or []

    def get_available_skills(self):
        return self._call("get_available_skills") or []

    def get_skill_tools(self):
        """{active skill: [the tools it declares]}."""
        return self._call("get_skill_tools") or {}

    def get_tool_processes(self):
        """{process: {kind, sandbox, tools}} for the selected tools - the MCP
        servers and the python tool, each with the sandbox it runs behind."""
        return self._call("get_tool_processes") or {}

    def set_selected_skills(self, names):
        self._call("set_selected_skills", names)

    def drain_errors(self):
        """the errors the agent recovered from (a tool that failed to load,
        ...) and has not reported yet; reading clears them."""
        errors = self._call("drain_errors") or []
        with self._ctrl_lock:
            failed = self._failed
            self._failed = []
        return failed + errors

    def set_model(self, model):
        self._call("set_model", model)

    def get_paths(self):
        """the live path policy: {"allowed": [...], "disallowed": [...]}."""
        return self._call("get_paths") or {}

    def set_paths(self, allowed=None, disallowed=None):
        """change the path policy: each side is the flag's comma list ("" clears),
        None leaves it alone. returns the error text, or None when it went."""
        value = {}
        value["allowed"] = allowed
        value["disallowed"] = disallowed
        with self._ctrl_lock:
            ok, result, error = self._ctrl.control("set_paths", value)
        if ok:
            return None
        return error or "set_paths failed"

    def set_system_prompt_base(self, base):
        self._call("set_system_prompt_base", base)

    def save(self, path):
        return self._call("save", path)

    def clone(self, spec=None):
        """swap the served agent for a fresh branch of itself (the 'clone' op).
        the socket and this client stay put - only the agent behind them
        changes. spec is an optional overrides dict of the branch's nameable
        state (model, system_prompt, messages, tools, skills,
        reasoning_effort, temperature, max_steps): a key present overrides -
        even with a None/empty value - and an absent key inherits, so
        clone({'messages': [...]}) is 'replace with fresh, seeded with these'.
        returns the branch's info dict ({'name': ...}), or None on failure."""
        return self._call("clone", spec)

    def load(self, path):
        with self._ctrl_lock:
            ok, _result, error = self._ctrl.control("load", path)
        if not ok:
            self._control_failed("load", error)
        return ok

    def close(self):
        try:
            self.stream.channel.close()
        except OSError:
            pass
        try:
            self._ctrl.channel.close()
        except OSError:
            pass


class ScreenUI(BaseUI):
    """the TUI client's UI: renders a served hook's prompts against the Screen.

    one-way ops draw immediately (notify -> a transcript line, status -> the
    status line). the input ops bridge to the main thread via
    Screen.submit_request, which services the request from inside prompt() by
    running the matching overlay and handing back the human's answer; a closed
    screen / no UI yields the headless default."""

    def __init__(self, screen, status, transcript):
        self._screen = screen
        self._status = status
        self._transcript = transcript

    def confirm(self, message, *, default=False, detail=""):
        request = {}
        request["kind"] = "confirm"
        request["title"] = message
        request["body"] = detail
        result = self._screen.submit_request(request)
        if result is None:
            return default
        return bool(result)

    def select(self, message, options, *, default=None, detail=""):
        options = list(options)
        request = {}
        request["kind"] = "select"
        request["title"] = message
        request["options"] = options
        result = self._screen.submit_request(request)
        if result is None:
            return BaseUI.select(self, message, options, default=default, detail=detail)
        return result

    def multiselect(self, message, options, *, default=(), detail=""):
        options = list(options)
        request = {}
        request["kind"] = "multiselect"
        request["title"] = message
        request["options"] = options
        request["default"] = list(default)
        result = self._screen.submit_request(request)
        if result is None:
            return BaseUI.multiselect(self, message, options, default=default, detail=detail)
        return result

    def text(self, message, *, default="", secret=False):
        request = {}
        request["kind"] = "text"
        request["title"] = message
        request["default"] = default
        request["secret"] = secret
        return self._screen.submit_request(request)

    def notify(self, message, *, level="info"):
        kind = Screen.META
        if level == "error":
            kind = Screen.ERROR
        self._transcript.note(f"[{level}] {message}\n", kind)

    def status(self, message):
        self._status.set_note(message)


class _Jobs:
    """the prompts queued behind the running turn, oldest first - a
    queue.Queue the :pending view can also list and remove from. put() from
    any thread; get(timeout) blocks like Queue.get and raises queue.Empty. an
    entry is a prompt string, the _CONTINUE sentinel, or None (stop)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._items = []

    def put(self, item):
        with self._cond:
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout=None):
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            if not self._items:
                raise queue.Empty
            return self._items.pop(0)

    def snapshot(self):
        """the queued entries in run order (a copy)."""
        with self._cond:
            return list(self._items)

    def remove(self, index, item):
        """drop the entry at index if it still is `item` (the worker taking the
        head shifts the list under a stale listing); returns whether it did."""
        with self._cond:
            if index < 0 or index >= len(self._items):
                return False
            if self._items[index] is not item and self._items[index] != item:
                return False
            del self._items[index]
            return True


class _Worker(threading.Thread):
    """pumps the agent's stream into the screen, all over the wire client.

    the served agent broadcasts every turn - whoever submitted it - as EVENTs
    closed by a RESULT, so the worker drains the stream wire at all times, not
    only during a turn it started: a prompt or steer sent by another client on
    the socket renders here exactly like a local one (the host emits the USER
    event at turn start). busy is a fact of the wire: the first event of a turn
    while idle marks the screen and status busy, the RESULT marks them idle
    again. a local prompt is submitted only when nothing is in flight, so the
    jobs queue serialises runs and never drives the agent concurrently."""

    def __init__(self, client, screen, jobs, stop, status, settings, pending):
        super().__init__(daemon=True, name="cai-tui-worker")
        self._client = client
        self._screen = screen
        self._jobs = jobs
        self._stop_event = stop
        self._status = status
        self._pending = pending
        self._transcript = _Transcript(screen, settings)
        self._ui = ScreenUI(screen, status, self._transcript)
        self._sample_tokens = 0
        self._sample_chars = 0
        self._interrupted = False
        self._in_flight = False

    @property
    def ui(self):
        """the UI the agent's questions are answered through on this
        worker's screen - what a `:`-command runs under (cai.ui.set_ui)."""
        return self._ui

    def mark_interrupted(self):
        """flag the in-flight run as Ctrl-C'd so the RESULT notes it once the
        run unwinds. set from the key thread; read on the worker thread."""
        self._interrupted = True

    def run(self):
        while not self._stop_event.is_set():
            if not self._in_flight:
                if not self._take_job():
                    break
            if not self._pump():
                break

    def _take_job(self):
        """submit the next queued local prompt, if any. False on the stop
        sentinel."""
        try:
            text = self._jobs.get(timeout=0)
        except queue.Empty:
            return True
        if text is None:
            return False
        try:
            if text is _CONTINUE:
                self._client.continue_run()
            else:
                # a real user turn was counted as pending when it entered the
                # queue; now that it starts running it is in flight, not pending.
                # _CONTINUE turns were never counted, so they never decrement.
                self._pending.user_started()
                self._client.submit(text)
        except OSError as e:
            self._transcript.note(f"[error: {e}]\n", Screen.ERROR)
            return True
        # busy at once, before the first event lands, so the steer / Ctrl-C
        # gates see the turn from the moment it was sent.
        self._begin()
        return True

    def _pump(self):
        """wait up to _PUMP_TIMEOUT for the stream and render what arrived.
        False once the wire is gone (EOF or a dead socket)."""
        try:
            readable, _, _ = select.select([self._client.stream.channel], [], [], _PUMP_TIMEOUT)
        except (OSError, ValueError):
            readable = None
        if readable is None:
            self._lost(None)
            return False
        if not readable:
            return True
        try:
            messages = self._client.recv()
        except OSError as e:
            self._lost(e)
            return False
        if messages is None:
            self._lost(None)
            return False
        for msg in messages:
            self._handle_message(msg)
        return True

    # not `_handle`: python 3.13's threading.Thread keeps its own thread handle
    # in an attribute of that name, which would replace this method.
    def _handle_message(self, msg):
        if self._client.stream.answer(msg, self._ui):
            return
        kind = msg.get("type")
        if kind == Wire.EVENT:
            self._begin()
            event = Wire.event_from_dict(msg["event"])
            self._transcript.event(event)
            self._update_status(event)
            return
        if kind == Wire.RESULT:
            self._end(msg.get("text"))

    def _begin(self):
        if self._in_flight: return
        self._in_flight = True
        self._interrupted = False
        self._screen.set_busy(True)
        self._status.busy()

    def _end(self, result):
        # the turn is over: the last block gets its final render before any
        # closing note, and the next turn starts on a fresh block.
        self._transcript.end()
        if result and result.startswith("Error:"):
            self._transcript.note(f"[{result}]\n", Screen.ERROR)
        if self._interrupted:
            self._transcript.note("[interrupted]\n", Screen.META)
        self._in_flight = False
        self._screen.set_busy(False)
        self._status.idle()

    def _lost(self, error):
        """the stream is gone: close out an in-flight turn and, unless we are
        shutting down anyway, say so."""
        if self._in_flight:
            self._end(None)
        if error is not None and not self._stop_event.is_set():
            self._transcript.note(f"[error: {error}]\n", Screen.ERROR)

    def _update_status(self, event):
        if event.type == EventType.CONTENT:
            self._status.stream("responding")
        elif event.type == EventType.REASONING:
            self._status.stream("reasoning")
        elif event.type == EventType.TOOL_CALL:
            self._status.tool()
        elif event.type == EventType.TOOL_RESULT:
            self._status.tool_done()
        elif event.type == EventType.USAGE:
            self._on_usage(event)

    def _on_usage(self, event):
        tokens = usage.total_tokens(event.usage)
        if not tokens: return
        # the api's exact count - the official readout. also kept as the
        # calibration sample the :messages overlay uses for its per-message math.
        messages = self._client.get_messages()
        self._status.set_tokens(tokens)
        self._sample_tokens = tokens
        self._sample_chars = usage.message_chars(messages)
        self._status.set_sample(self._sample_tokens, self._sample_chars)


def _agent_control(name, op):
    """connect to an agent's socket and run one read-only control op; returns its
    value, or None if the socket is gone/unreachable. the :agents view reads
    everything this way - solely over ~/.config/cai/agents/*.sock."""
    from cai.agents_registry import AgentsRegistry

    try:
        channel = AgentsRegistry.connect(name)
    except OSError:
        return None
    wire = Wire(channel)
    try:
        ok, value, _error = wire.control(op)
    except OSError:
        ok, value = False, None
    finally:
        channel.close()
    if not ok:
        return None
    return value
