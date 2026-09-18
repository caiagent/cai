"""Tests for the TUI worker pumping the stream wire at all times: a turn that
another client submits or steers on the served agent's socket renders in the
live conversation view, and busy/idle follows the wire rather than the local
jobs queue."""
import threading
import time

from test_wired_agent import FakeApi, make_agent
from cai import channel
from cai.environment import Settings
from cai.events import EventType
from cai.tui import AgentClient, _Jobs, _Worker
from cai.wire import Wire
from cai.wired_agent import UnixWiredAgent


class FakeScreen:
    def __init__(self):
        self.writes = []
        self._busy = False
        self.busy_log = []

    def write(self, text, kind=None, block=False):
        self.writes.append((text, kind))

    def set_busy(self, busy):
        self._busy = busy
        self.busy_log.append(busy)

    def submit_request(self, request):
        return None


class FakeStatus:
    def __init__(self):
        self.log = []

    def busy(self):
        self.log.append("busy")

    def idle(self):
        self.log.append("idle")

    def stream(self, kind):
        self.log.append(kind)

    def tool(self):
        self.log.append("tool")

    def tool_done(self):
        self.log.append("tool_done")

    def set_tokens(self, tokens):
        pass

    def set_sample(self, tokens, chars):
        pass

    def set_note(self, message):
        pass


class FakePending:
    def __init__(self):
        self.started = 0

    def user_started(self):
        self.started += 1


def _text(screen):
    out = []
    for text, kind in screen.writes:
        out.append(text)
    return "".join(out)


def _wait(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _drain_result(wire):
    while True:
        messages = wire.recv()
        for msg in messages:
            if msg["type"] == Wire.RESULT:
                return msg["text"]


def _serve(tmp_path, api):
    path = str(tmp_path / "a.sock")
    served = UnixWiredAgent(make_agent(api=api), path)
    thread = threading.Thread(target=served.serve, daemon=True)
    thread.start()
    return served, thread


def _worker(client, screen, status, stop, pending):
    jobs = _Jobs()
    worker = _Worker(client, screen, jobs, stop, status, Settings(), pending)
    worker.start()
    return worker, jobs


def test_remote_submit_renders_in_the_live_view(tmp_path):
    served, thread = _serve(tmp_path, FakeApi(chunks=["hello ", "world"]))
    client = AgentClient(served.path)
    screen = FakeScreen()
    status = FakeStatus()
    stop = threading.Event()
    pending = FakePending()
    worker, jobs = _worker(client, screen, status, stop, pending)
    try:
        other = Wire(channel.connect(served.path))
        other.send_submit("from afar")
        assert _drain_result(other) == "hello world"
        assert _wait(lambda: screen.busy_log[-1:] == [False])
        text = _text(screen)
        assert "> from afar" in text
        assert "hello world" in text
        assert screen.busy_log == [True, False]
        assert status.log[0] == "busy"
        assert status.log[-1] == "idle"
        assert pending.started == 0        # no local job was involved
        other.channel.close()
    finally:
        stop.set()
        jobs.put(None)
        worker.join(timeout=5)
        client.close()
        served.close()
        thread.join(timeout=5)


def test_remote_steer_while_idle_renders_in_the_live_view(tmp_path):
    served, thread = _serve(tmp_path, FakeApi(chunks=["steered"]))
    client = AgentClient(served.path)
    screen = FakeScreen()
    status = FakeStatus()
    stop = threading.Event()
    worker, jobs = _worker(client, screen, status, stop, FakePending())
    try:
        other = Wire(channel.connect(served.path))
        other.send_steer("nudge")
        assert _drain_result(other) == "steered"
        assert _wait(lambda: screen.busy_log[-1:] == [False])
        assert "steered" in _text(screen)
        other.channel.close()
    finally:
        stop.set()
        jobs.put(None)
        worker.join(timeout=5)
        client.close()
        served.close()
        thread.join(timeout=5)


def test_local_turn_after_a_remote_one_is_not_confused_by_stale_output(tmp_path):
    served, thread = _serve(tmp_path, FakeApi(chunks=["answer"]))
    client = AgentClient(served.path)
    screen = FakeScreen()
    status = FakeStatus()
    stop = threading.Event()
    pending = FakePending()
    worker, jobs = _worker(client, screen, status, stop, pending)
    try:
        other = Wire(channel.connect(served.path))
        other.send_submit("remote")
        assert _drain_result(other) == "answer"
        assert _wait(lambda: screen.busy_log == [True, False])
        jobs.put("local")
        assert _wait(lambda: screen.busy_log == [True, False, True, False])
        assert pending.started == 1
        text = _text(screen)
        assert text.index("> remote") < text.index("> local")
        assert text.count("answer") == 2
        assert client.get_messages()[-1]["content"] == "answer"
        other.channel.close()
    finally:
        stop.set()
        jobs.put(None)
        worker.join(timeout=5)
        client.close()
        served.close()
        thread.join(timeout=5)


def test_worker_exits_when_the_stream_closes(tmp_path):
    served, thread = _serve(tmp_path, FakeApi())
    client = AgentClient(served.path)
    screen = FakeScreen()
    stop = threading.Event()
    worker, jobs = _worker(client, screen, FakeStatus(), stop, FakePending())
    served.close()
    thread.join(timeout=5)
    worker.join(timeout=5)
    assert not worker.is_alive()
    stop.set()
    client.close()
