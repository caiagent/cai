"""request_records: one HTTP request to the model server, written down.

A RequestRecord is what went out (url, headers, body) and what came back
(status, headers, the assembled answer, usage, or the error) for a single
chat request. The agentic loop makes one per request and hands it to the api
layer, which fills it in as the request progresses; once the request ended -
answered, failed or interrupted - the loop hands the record to the
after_request hooks (ctx.request) and keeps nothing itself.

So recording is entirely up to whoever registers a hook: the TUI keeps the
recent ones in memory for its :requests view (cai.tui.requests), an extension
could log them to a file, and an SDK or headless run with no such hook pays
for one small object per request and nothing more.

A record is cheap by design: it holds a reference to the request body the api
already built (never a copy), and a streamed answer is stored assembled, not
as its raw SSE lines."""
import time

from urllib.parse import urlsplit

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
    """one HTTP request and what came back. it starts out empty; the api layer
    fills it in as the request progresses: sent() when it goes out,
    responded() when the status line arrives, then exactly one of finished() /
    failed() / aborted()."""

    def __init__(self):
        self.id = None                  # for whoever keeps records to assign
        self.sent("", "", {}, {})

    def sent(self, method, url, headers, body):
        """the request is going out now. also starts the record over, for an
        api call that sends the same request again (its own retries)."""
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
