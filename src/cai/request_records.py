"""request_records: the HTTP requests an agent sent to the model server.

Every chat request the api layer makes is written down as a RequestRecord -
what went out (url, headers, body) and what came back (status, headers, the
assembled answer, usage, or the error) - and kept in the agent's
RequestRecords, a short in-memory list of the most recent ones. The TUI's
:requests view reads them over the wire; nothing here is ever written to disk,
and the list dies with the agent.

It is a troubleshooting aid, so it is kept cheap: the list is bounded, a
record holds a reference to the request body the api already built (never a
copy), and a streamed answer is stored assembled, not as its raw SSE lines.

Each agent owns its own RequestRecords and hands it to the api per call, so
agents that share one api client never see each other's requests."""
import threading
import time

from collections import deque
from urllib.parse import urlsplit

# how many requests an agent remembers; older ones fall off the front.
LIMIT = 50

# how much of a failed response's body a record keeps.
_ERROR_BODY_MAX = 4000

# a record's life: sent -> (headers arrived) -> one of the three endings.
PENDING = "pending"        # sent, nothing back yet
RECEIVING = "receiving"    # a 200 came back, the body is still being read
DONE = "done"              # the whole answer was read
FAILED = "failed"          # a network error, a non-200, or an unusable body
ABORTED = "aborted"        # cut short by an interrupt

_ENDED = (DONE, FAILED, ABORTED)


def redact_headers(headers):
    """a copy of request headers safe to show: the Authorization value keeps
    its scheme ('Bearer') and loses the key."""
    shown = {}
    for name, value in headers.items():
        if name.lower() == "authorization":
            scheme = str(value).split(" ", 1)[0]
            value = f"{scheme} ***"
        shown[name] = value
    return shown


class RequestRecord:
    """one HTTP request and what came back. the api layer fills it in as the
    request progresses: responded() when the status line arrives, then exactly
    one of finished() / failed() / aborted()."""

    def __init__(self, id, method, url, headers, body):
        self.id = id
        self.method = method
        self.url = url
        self.request_headers = redact_headers(headers)
        self.body = body                # the dict the api sent, by reference
        self.started = time.time()      # wall clock, for display
        self._clock = time.monotonic()  # for the durations

        self.state = PENDING
        self.status = None              # HTTP status; None until one arrives
        self.response_headers = {}
        self.first_byte = None          # seconds until the status line arrived
        self.duration = None            # seconds until the record ended
        self.error = None

        self.content = ""
        self.reasoning = ""
        self.tool_calls = None
        self.usage = {}
        self.chunks = 0                 # SSE chunks read (streaming only)

    def _elapsed(self):
        return time.monotonic() - self._clock

    def responded(self, status, headers):
        """the status line and headers arrived."""
        self.status = status
        self.response_headers = dict(headers)
        self.first_byte = self._elapsed()
        self.state = RECEIVING

    def finished(self, content, reasoning, tool_calls, usage):
        """the whole answer was read."""
        self.content = content or ""
        self.reasoning = reasoning or ""
        self.tool_calls = tool_calls
        self.usage = usage or {}
        self.duration = self._elapsed()
        self.state = DONE

    def failed(self, error):
        """the request ended in an error: no response at all, a non-200 (call
        responded() first so the status and headers are kept), or a body that
        could not be used."""
        self.error = str(error)[:_ERROR_BODY_MAX]
        self.duration = self._elapsed()
        self.state = FAILED

    def aborted(self):
        """the request was cut short. does nothing to a record that already
        ended, so it is safe to call from a `finally`."""
        if self.state in _ENDED:
            return
        self.duration = self._elapsed()
        self.state = ABORTED

    def summary(self):
        """the few fields the :requests overview row shows - small enough to
        send for every record on every refresh."""
        body = self.body or {}
        out = {}
        out["id"] = self.id
        out["started"] = self.started
        out["state"] = self.state
        out["method"] = self.method
        out["path"] = urlsplit(self.url).path
        out["model"] = body.get("model")
        out["stream"] = bool(body.get("stream"))
        out["messages"] = len(body.get("messages") or [])
        out["tools"] = len(body.get("tools") or [])
        out["status"] = self.status
        out["duration"] = self.duration
        out["first_byte"] = self.first_byte
        out["prompt_tokens"] = self.usage.get("prompt_tokens")
        out["completion_tokens"] = self.usage.get("completion_tokens")
        out["retry_after"] = self.response_headers.get("Retry-After")
        out["error"] = self.error
        return out

    def detail(self):
        """everything about the request, for the expanded row and the editor."""
        request = {}
        request["method"] = self.method
        request["url"] = self.url
        request["headers"] = self.request_headers
        request["body"] = self.body

        response = {}
        response["status"] = self.status
        response["headers"] = self.response_headers
        response["first_byte"] = self.first_byte
        response["duration"] = self.duration
        response["chunks"] = self.chunks
        response["content"] = self.content
        response["reasoning"] = self.reasoning
        response["tool_calls"] = self.tool_calls
        response["usage"] = self.usage
        response["error"] = self.error

        out = {}
        out["id"] = self.id
        out["started"] = self.started
        out["state"] = self.state
        out["request"] = request
        out["response"] = response
        return out


class RequestRecords:
    """the most recent requests of one agent, oldest first. the api's request
    thread adds and fills records while the wire's control thread reads them,
    so the list itself is guarded by a lock."""

    def __init__(self, limit=LIMIT):
        self._records = deque(maxlen=limit)
        self._lock = threading.Lock()
        self._next_id = 1

    def open(self, method, url, headers, body):
        """start the record of a request that is about to be sent."""
        with self._lock:
            record = RequestRecord(self._next_id, method, url, headers, body)
            self._next_id += 1
            self._records.append(record)
        return record

    def summaries(self):
        """summary() of every remembered request, oldest first."""
        with self._lock:
            records = list(self._records)
        summaries = []
        for record in records:
            summaries.append(record.summary())
        return summaries

    def detail(self, id):
        """detail() of the request with this id, or None once it fell off."""
        with self._lock:
            records = list(self._records)
        for record in records:
            if record.id != id: continue
            return record.detail()
        return None
