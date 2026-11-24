"""Tests for the :requests feature: the record the api layer fills in
(cai.request_records), the hooks the loop announces it with, the TUI store
that keeps them (cai.tui.requests), and the overlay that shows them. Offline -
requests.post is the scripted fake from test_api."""
import os
import pty
import threading
import time

import pytest
import requests

from test_api import FakeResponse, blocking_body, client, install, sse
from test_wired_agent import make_agent

from cai.api import ApiError
from cai.environment import Settings
from cai.hooks import HookEvent, HooksRegistry
from cai.llm import call_llm
from cai.request_records import RequestRecord, redact_headers
from cai.screen.ansi import KEY_ESC, ansi_strip
from cai.screen.overlays import requests as overlay
from cai.tui.requests import RequestStore


def test_authorization_is_redacted():
    shown = redact_headers({"Authorization": "Bearer sk-secret", "Content-Type": "application/json"})
    assert shown == {"Authorization": "Bearer ***", "Content-Type": "application/json"}


def test_blocking_request_is_recorded(monkeypatch):
    usage = {"prompt_tokens": 12, "completion_tokens": 3}
    install(monkeypatch, FakeResponse(body=blocking_body(content="ok", usage=usage),
                                      headers={"x-request-id": "abc"}))
    record = RequestRecord()
    client().chat([{"role": "user", "content": "hi"}], "m", record=record)

    summary = record.summary()
    assert summary["state"] == "done"
    assert summary["status"] == 200
    assert summary["method"] == "POST"
    assert summary["path"] == "/v1/chat/completions"
    assert summary["model"] == "m"
    assert summary["messages"] == 1
    assert summary["prompt_tokens"] == 12
    assert summary["completion_tokens"] == 3

    detail = record.detail()
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
    record = RequestRecord()
    list(client().chat([], "m", stream=True, record=record))

    detail = record.detail()
    assert detail["state"] == "done"
    assert detail["response"]["content"] == "hello"
    assert detail["response"]["reasoning"] == "hm"
    assert detail["response"]["chunks"] == 4
    assert detail["response"]["usage"]["completion_tokens"] == 2


def test_failed_request_is_recorded(monkeypatch):
    install(monkeypatch, FakeResponse(status_code=429, headers={"Retry-After": "7"}))
    record = RequestRecord()
    with pytest.raises(ApiError):
        client().chat([], "m", retry=False, record=record)
    summary = record.summary()
    assert (summary["state"], summary["status"], summary["retry_after"]) == ("failed", 429, "7")
    assert summary["error"] == "error-body"

    install(monkeypatch, exc=requests.ConnectionError("down"))
    record = RequestRecord()
    with pytest.raises(ApiError):
        client().chat([], "m", retry=False, record=record)
    summary = record.summary()
    assert (summary["state"], summary["status"]) == ("failed", None)
    assert "down" in summary["error"]


def test_unusable_body_is_recorded_as_failed(monkeypatch):
    install(monkeypatch, FakeResponse(raise_json=True))
    record = RequestRecord()
    with pytest.raises(ApiError):
        client().chat([], "m", record=record)
    summary = record.summary()
    assert summary["state"] == "failed"
    assert summary["status"] == 200
    assert "invalid JSON" in summary["error"]


def test_abandoned_stream_is_recorded_as_aborted(monkeypatch):
    lines = [sse({"choices": [{"delta": {"content": "a"}}]}),
             sse({"choices": [{"delta": {"content": "b"}}]}),
             b"data: [DONE]"]
    install(monkeypatch, FakeResponse(lines=lines))
    record = RequestRecord()
    gen = client().chat([], "m", stream=True, record=record)
    next(gen)
    gen.close()
    assert record.summary()["state"] == "aborted"


# --------------------------------------------------------------------------
# the hooks the loop announces each request with
# --------------------------------------------------------------------------

def test_loop_fires_after_request_for_every_attempt(monkeypatch):
    script = [FakeResponse(status_code=429, headers={"Retry-After": "7"}),
              FakeResponse(body=blocking_body(content="ok"))]
    install(monkeypatch, script=script)
    monkeypatch.setattr("cai.api.time.sleep", lambda seconds: None)
    seen = []

    def after(ctx):
        seen.append(ctx.request)

    hooks = HooksRegistry.from_list([(HookEvent.AFTER_REQUEST, after)])
    list(call_llm([{"role": "user", "content": "hi"}], "m", client(),
                  hooks=hooks, stream=False))

    # one complete record per attempt.
    limited, good = seen
    assert limited is not good
    assert (limited.state, limited.status) == ("failed", 429)
    assert limited.body["messages"] == [{"role": "user", "content": "hi"}]
    assert (good.state, good.content) == ("done", "ok")


def test_after_request_fires_for_a_request_that_fails_for_good(monkeypatch):
    install(monkeypatch, FakeResponse(status_code=400))
    ended = []
    hooks = HooksRegistry.from_list(
        [(HookEvent.AFTER_REQUEST, lambda ctx: ended.append(ctx.request.status))])
    with pytest.raises(ApiError):
        list(call_llm([{"role": "user", "content": "hi"}], "m", client(),
                      hooks=hooks, stream=False))
    assert ended == [400]


# --------------------------------------------------------------------------
# the TUI's store
# --------------------------------------------------------------------------

def _streamed_ok():
    """a streaming response whose whole answer is 'ok' (an Agent run streams)."""
    lines = [sse({"choices": [{"delta": {"content": "ok"}}]}), b"data: [DONE]"]
    return FakeResponse(lines=lines)


def _stored_agent(api, store, name):
    """a bare test agent reporting its requests to `store`, the way the TUI
    sets its agents up."""
    agent = make_agent(api=api, hooks=[(HookEvent.AFTER_REQUEST, store.on_request)])
    agent.name = name
    return agent


def test_an_agent_keeps_no_requests_itself(monkeypatch):
    install(monkeypatch, _streamed_ok())
    agent = make_agent(api=client())
    list(agent.run("hi"))
    assert not hasattr(agent, "request_records")
    assert not hasattr(agent, "get_requests")


def test_store_files_each_request_under_its_agent(monkeypatch):
    install(monkeypatch, _streamed_ok())
    shared = client()
    store = RequestStore(Settings())
    one = _stored_agent(shared, store, "one")
    two = _stored_agent(shared, store, "two")
    list(one.run("hi"))
    list(two.run("hi"))
    list(one.run("again"))

    assert [s["id"] for s in store.summaries("one")] == [1, 3]
    assert [s["id"] for s in store.summaries("two")] == [2]
    assert store.summaries("nobody") == []
    assert store.detail("one", 3)["response"]["content"] == "ok"
    assert store.detail("two", 3) is None       # another agent's request


def test_store_keeps_only_the_most_recent(monkeypatch):
    install(monkeypatch, _streamed_ok())
    settings = Settings()
    settings.keep_requests = 2
    store = RequestStore(settings)
    agent = _stored_agent(client(), store, "one")
    for _ in range(4):
        list(agent.run("hi"))
    assert [s["id"] for s in store.summaries("one")] == [3, 4]
    assert store.detail("one", 1) is None


def test_store_keeps_nothing_when_the_setting_is_zero(monkeypatch):
    install(monkeypatch, _streamed_ok())
    settings = Settings()
    settings.keep_requests = 0
    store = RequestStore(settings)
    agent = _stored_agent(client(), store, "one")
    list(agent.run("hi"))
    assert store.summaries("one") == []


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


def test_summary_line_for_requests_that_did_not_succeed():
    limited = _summary(state="failed", status=429, retry_after="7", error="slow down\nmore")
    line = overlay.summary_line(limited)
    assert "retry-after 7" in line
    assert "slow down" in line and "more" not in line
    assert overlay.status_label(limited) == "429"

    assert overlay.status_label(_summary(state="failed", status=None)) == "ERR"

    cut = _summary(state="aborted", status=None)
    assert overlay.status_label(cut) == "CUT"
    assert "interrupted" in overlay.summary_line(cut)


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
