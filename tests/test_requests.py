"""Tests for the :requests feature: the records the api layer writes down
(cai.request_records), their trip over the wire, and the overlay that shows
them. Offline - requests.post is the scripted fake from test_api."""
import os
import pty
import threading
import time

import pytest
import requests

from test_api import FakeResponse, blocking_body, client, install, sse
from test_wired_agent import make_agent

from cai.api import ApiError
from cai.request_records import RequestRecords, redact_headers
from cai.screen.ansi import KEY_ESC, ansi_strip
from cai.screen.overlays import requests as overlay
from cai.wired_agent import WiredAgent


def test_authorization_is_redacted():
    shown = redact_headers({"Authorization": "Bearer sk-secret", "Content-Type": "application/json"})
    assert shown == {"Authorization": "Bearer ***", "Content-Type": "application/json"}


def test_blocking_request_is_recorded(monkeypatch):
    usage = {"prompt_tokens": 12, "completion_tokens": 3}
    install(monkeypatch, FakeResponse(body=blocking_body(content="ok", usage=usage),
                                      headers={"x-request-id": "abc"}))
    records = RequestRecords()
    client().chat([{"role": "user", "content": "hi"}], "m", records=records)

    [summary] = records.summaries()
    assert summary["state"] == "done"
    assert summary["status"] == 200
    assert summary["method"] == "POST"
    assert summary["path"] == "/v1/chat/completions"
    assert summary["model"] == "m"
    assert summary["messages"] == 1
    assert summary["prompt_tokens"] == 12
    assert summary["completion_tokens"] == 3

    detail = records.detail(summary["id"])
    assert detail["request"]["headers"]["Authorization"] == "Bearer ***"
    assert "sk-test" not in str(detail)
    assert detail["request"]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert detail["response"]["headers"] == {"x-request-id": "abc"}
    assert detail["response"]["content"] == "ok"
    assert detail["response"]["usage"] == usage


def test_streamed_answer_is_recorded_assembled(monkeypatch):
    lines = [sse({"choices": [{"delta": {"reasoning": "hm"}}]}),
             sse({"choices": [{"delta": {"content": "he"}}]}),
             sse({"choices": [{"delta": {"content": "llo"}}]}),
             sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
             b"data: [DONE]"]
    install(monkeypatch, FakeResponse(lines=lines))
    records = RequestRecords()
    list(client().chat([], "m", stream=True, records=records))

    detail = records.detail(records.summaries()[0]["id"])
    assert detail["state"] == "done"
    assert detail["response"]["content"] == "hello"
    assert detail["response"]["reasoning"] == "hm"
    assert detail["response"]["chunks"] == 4
    assert detail["response"]["usage"]["completion_tokens"] == 2


def test_every_attempt_gets_its_own_record(monkeypatch):
    script = [FakeResponse(status_code=429, headers={"Retry-After": "7"}),
              requests.ConnectionError("down"),
              FakeResponse(body=blocking_body(content="ok"))]
    install(monkeypatch, script=script)
    monkeypatch.setattr("cai.api.time.sleep", lambda seconds: None)
    records = RequestRecords()
    client().chat([], "m", records=records)

    limited, dropped, good = records.summaries()
    assert (limited["state"], limited["status"], limited["retry_after"]) == ("failed", 429, "7")
    assert limited["error"] == "error-body"
    assert (dropped["state"], dropped["status"]) == ("failed", None)
    assert "down" in dropped["error"]
    assert (good["state"], good["status"]) == ("done", 200)


def test_unusable_body_is_recorded_as_failed(monkeypatch):
    install(monkeypatch, FakeResponse(raise_json=True))
    records = RequestRecords()
    with pytest.raises(ApiError):
        client().chat([], "m", records=records)
    [summary] = records.summaries()
    assert summary["state"] == "failed"
    assert summary["status"] == 200
    assert "invalid JSON" in summary["error"]


def test_abandoned_stream_is_recorded_as_aborted(monkeypatch):
    lines = [sse({"choices": [{"delta": {"content": "a"}}]}),
             sse({"choices": [{"delta": {"content": "b"}}]}),
             b"data: [DONE]"]
    install(monkeypatch, FakeResponse(lines=lines))
    records = RequestRecords()
    gen = client().chat([], "m", stream=True, records=records)
    next(gen)
    gen.close()
    assert records.summaries()[0]["state"] == "aborted"


def _streamed_ok():
    """a streaming response whose whole answer is 'ok' (an Agent run streams)."""
    lines = [sse({"choices": [{"delta": {"content": "ok"}}]}), b"data: [DONE]"]
    return FakeResponse(lines=lines)


def test_only_the_most_recent_requests_are_kept():
    records = RequestRecords(limit=3)
    for _ in range(5):
        records.open("POST", "https://example.test/v1/x", {}, {})
    assert [s["id"] for s in records.summaries()] == [3, 4, 5]
    assert records.detail(1) is None
    assert records.detail(5)["id"] == 5


def test_agents_sharing_an_api_keep_separate_records(monkeypatch):
    install(monkeypatch, _streamed_ok())
    shared = client()
    one = make_agent(api=shared)
    two = make_agent(api=shared)
    list(one.run("hi"))
    assert len(one.get_requests()) == 1
    assert two.get_requests() == []


def test_requests_are_served_over_the_wire(monkeypatch):
    install(monkeypatch, _streamed_ok())
    agent = make_agent(api=client())
    list(agent.run("hi"))
    served = WiredAgent(agent)

    ok, summaries, _error = served._control("get_requests", None)
    assert ok is True
    assert len(summaries) == 1
    ok, detail, _error = served._control("get_request", summaries[0]["id"])
    assert detail["response"]["content"] == "ok"
    assert served._control("get_request", 999) == (True, None, None)


# --------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------

def _summary(**changes):
    summary = {"id": 1, "started": 0, "state": "done", "method": "POST",
               "path": "/v1/chat/completions", "model": "m", "stream": True,
               "messages": 41, "tools": 2, "status": 200, "duration": 3.21,
               "first_byte": 0.8, "prompt_tokens": 18000, "completion_tokens": 620,
               "retry_after": None, "error": None}
    summary.update(changes)
    return summary


def _detail(**response):
    body = {"model": "m", "stream": True,
            "tools": [{"function": {"name": "fs__read"}}],
            "messages": [{"role": "user", "content": f"message {i}"} for i in range(10)]}
    reply = {"status": 200, "headers": {"x-ratelimit-remaining": "9"}, "first_byte": 0.8,
             "duration": 3.21, "chunks": 12, "content": "the answer", "reasoning": "",
             "tool_calls": None, "usage": {"prompt_tokens": 5}, "error": None}
    reply.update(response)
    return {"id": 1, "started": 0, "state": "done",
            "request": {"method": "POST", "url": "https://example.test/v1/chat/completions",
                        "headers": {"Authorization": "Bearer ***"}, "body": body},
            "response": reply}


def test_summary_line_shows_the_high_level_facts():
    line = overlay.summary_line(_summary())
    assert "POST /v1/chat/completions" in line
    assert "3.21s" in line
    assert "41 msgs" in line
    assert "18.0k→620 tok" in line
    assert overlay.status_label(_summary()) == "200"


def test_summary_line_for_a_rate_limited_and_a_pending_request():
    limited = _summary(state="failed", status=429, retry_after="7", error="slow down\nmore")
    line = overlay.summary_line(limited)
    assert "retry-after 7" in line
    assert "slow down" in line and "more" not in line
    assert overlay.status_label(limited) == "429"

    pending = _summary(state="pending", status=None, duration=None)
    assert "waiting for the server" in overlay.summary_line(pending)
    assert overlay.status_label(pending) == "..."
    assert overlay.status_label(_summary(state="failed", status=None)) == "ERR"


def test_detail_lines_digest_the_exchange():
    text = "\n".join(overlay.detail_lines(_detail()))
    assert "POST https://example.test/v1/chat/completions" in text
    assert "Authorization: Bearer ***" in text
    assert "tools (1): fs__read" in text
    assert "messages (10):" in text
    assert "4 earlier" in text                  # only the newest are listed
    assert "message 9" in text and "message 0" not in text
    assert "response: 200" in text
    assert "x-ratelimit-remaining: 9" in text
    assert "the answer" in text


def test_editor_text_is_the_whole_exchange_as_json():
    import json

    assert json.loads(overlay.editor_text(_detail())) == _detail()


class _TtyScreen:
    def __init__(self, tty_fd, rows, cols):
        self._tty_fd = tty_fd
        self._rows = rows
        self._cols = cols


def test_overlay_lists_folds_and_unfolds(capsys):
    summaries = [_summary(id=1), _summary(id=2, state="failed", status=429, retry_after="7")]

    def detail_fn(rid):
        detail = _detail()
        detail["id"] = rid
        return detail

    master, slave = pty.openpty()
    shown = {}

    def press():
        time.sleep(0.3)
        shown["folded"] = ansi_strip(capsys.readouterr().out)
        os.write(master, b"\t")
        time.sleep(0.3)
        shown["unfolded"] = ansi_strip(capsys.readouterr().out)
        os.write(master, KEY_ESC.encode())

    presser = threading.Thread(target=press, daemon=True)
    presser.start()
    try:
        overlay.prompt_requests_overlay(_TtyScreen(slave, 40, 120), lambda: summaries, detail_fn)
    finally:
        presser.join(timeout=5)
        os.close(master)
        os.close(slave)

    assert "Requests" in shown["folded"]
    assert "▶ " in shown["folded"]
    assert "retry-after 7" in shown["folded"]
    assert "request headers" not in shown["folded"]
    # the cursor opens on the newest request; Tab unfolds its digest.
    assert "▼ " in shown["unfolded"]
    assert "request headers:" in shown["unfolded"]
    assert "Authorization: Bearer ***" in shown["unfolded"]
