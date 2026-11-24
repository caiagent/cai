import json
import time
import queue
import socket
import logging
import warnings
import threading
import requests

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from cai.origin import envelope, origin_of
from cai.request_records import ABORTED, RequestRecords
from urllib.parse import urlsplit, urlunsplit
from urllib3.exceptions import InsecureRequestWarning

log = logging.getLogger("cai")

# base delay before the second attempt; each further attempt doubles it.
# module-level so a test can zero it out.
_RETRY_BACKOFF = 0.5

# the same, for a rate-limited (429) request: a limit resets over seconds to a
# minute, so the short transient backoff would only burn the attempts.
_RATE_LIMIT_BACKOFF = 5.0

# the longest a server's Retry-After is obeyed for; a longer ask is clamped so
# one header cannot park a run for an hour.
_RETRY_AFTER_MAX = 60.0

# how often an interruptible request re-checks its interrupt Event while the
# pump thread is blocked on the network.
_POLL_TICK = 0.2


class ApiError(Exception):
    """a request failed for good: a transport error, a bad HTTP status, or an
    unusable response body - after the transient cases (network errors, 429,
    5xx) were retried. `status` carries the HTTP status when one was received.

    this is the api layer's whole error surface: OpenAiApi.chat() and
    SystemOneApi.system_one() raise it instead of returning a sentinel, so a
    failed call can never read as the model answering with an empty string.

    `retryable` says the failure was a transient one that hit before any
    response body was read, so sending the same request again is safe and may
    work; `retry_after` is the wait the server asked for (seconds, already
    clamped), when it sent one. OpenAiApi.retry_wait() turns the two into the
    next backoff - what a caller that passed retry=False drives its own retry
    loop with."""

    def __init__(self, message, status=None, retryable=False, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


def _retryable(status):
    """whether a failed request is worth retrying: any network-level failure
    (no status), rate limiting (429), or a server-side error (5xx). other 4xx
    are permanent - a retry cannot fix bad auth or a bad request."""
    if status is None:
        return True
    if status == 429:
        return True
    return status >= 500


def _parse_retry_after(value):
    """seconds to wait per a Retry-After header value - either delta-seconds
    or an HTTP-date - or None when absent/unparseable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if seconds != seconds:              # nan
        return None
    return max(0.0, seconds)


def _asked_wait(headers):
    """the wait a response's Retry-After header asks for, in seconds and
    clamped to _RETRY_AFTER_MAX, or None when it carries none."""
    asked = _parse_retry_after(headers.get('Retry-After'))
    if asked is None:
        return None
    return min(asked, _RETRY_AFTER_MAX)


def _retry_delay(status, attempt, retry_after=None):
    """seconds to wait after failed attempt number `attempt` (1-based). the
    wait the server asked for wins; otherwise a doubling backoff, from a
    longer base when the failure was a 429."""
    if retry_after is not None:
        return retry_after
    base = _RATE_LIMIT_BACKOFF if status == 429 else _RETRY_BACKOFF
    return base * (2 ** (attempt - 1))


def _unusable(record, message):
    """a response came back but its body cannot be used: mark `record` failed
    (unless an interrupt already ended it) and build the ApiError to raise."""
    if record.state != ABORTED:
        record.failed(message)
    return ApiError(message)


def _price_per_mtok(value):
    """convert a per-token USD price (string or number) to USD per million
    tokens, or None when absent/unparseable. '0' is a real price (free)."""
    if value is None:
        return None
    try:
        per_token = float(value)
    except (TypeError, ValueError):
        return None
    return per_token * 1_000_000


def _wire_messages(messages):
    """strip local-only ("_"-prefixed) keys before a message goes on the
    wire - some strict providers reject unknown message fields. the single
    choke point every request passes through. a message carrying `_origin`
    (text the harness injected, not the user's - see cai.origin) goes out with
    its content wrapped in that origin's envelope; the stored message is left
    untouched."""
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue

        has_local_keys = False
        for key in message:
            if key.startswith('_'):
                has_local_keys = True
                break
        if not has_local_keys:
            cleaned.append(message)
            continue

        wire_message = {}
        for key, value in message.items():
            if key.startswith('_'): continue
            wire_message[key] = value
        origin = origin_of(message)
        if origin is not None and 'content' in wire_message:
            wire_message['content'] = envelope(origin, wire_message['content'])
        cleaned.append(wire_message)
    return cleaned


class _Flight:
    """shared state between an interruptible request's foreground poll loop
    and its pump thread: the queue the pump feeds, and the live response once
    the POST came back (so an abort can reach the socket underneath it)."""

    def __init__(self):
        self.queue = queue.Queue()
        self.response = None


def _abort_flight(flight):
    """tear down an in-flight request from the foreground side. shutdown() on
    the underlying socket (not close() - only shutdown reliably wakes a recv
    blocked in another thread) makes the pump's read raise at once instead of
    running out the read timeout. best-effort: no response yet, a fake without
    a raw socket, or an already-dead connection all just pass."""
    r = flight.response
    if r is None: return
    try:
        r.raw.connection.sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        r.close()
    except Exception:
        pass


class OpenAiApi:
    """A minimal OpenAI-compatible chat client: one HTTP POST to
    /chat/completions that returns the assistant's message. This is the bottom
    of the stack - it knows nothing about tools-as-code or sessions; it speaks
    the wire format, retries the transient failures, and raises ApiError for
    everything it cannot recover."""

    def __init__(self,
                 base_url,
                 api_key,
                 ssl_verify=True,
                 timeout=(10, 120),
                 retries=3,
                 rate_limit_retries=5):
        self.base_url = base_url
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        if not ssl_verify:
            # only mute InsecureRequestWarning when verification is actually off.
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        # (connect, read) seconds.
        if isinstance(timeout, (list, tuple)):
            timeout = tuple(timeout)
        self.timeout = timeout
        # total attempts per chat request (1 = no retries).
        self.retries = max(1, retries)
        # the same budget for a request that keeps coming back 429.
        self.rate_limit_retries = max(1, rate_limit_retries)

    def _request_data(self,
                      messages,
                      model,
                      system_prompt,
                      tools,
                      tool_choice,
                      reasoning_effort,
                      temperature):
        data = {}
        data['model'] = model
        data['messages'] = []
        if tools:
            data['tools'] = tools
            data['tool_choice'] = tool_choice
        if reasoning_effort:
            # send both shapes: OpenRouter reads `reasoning`,
            # vLLM/DeepSeek read `reasoning_effort`.
            data['reasoning'] = {"effort": reasoning_effort}
            data['reasoning_effort'] = reasoning_effort
        if temperature is not None:
            data['temperature'] = temperature

        if system_prompt:
            data['messages'].append(system_prompt)
        data['messages'].extend(_wire_messages(messages))
        return data

    def chat(self,
             messages,
             model,
             system_prompt=None,
             tools=None,
             tool_choice="auto",
             reasoning_effort=None,
             temperature=None,
             stream=False,
             interrupt=None,
             retry=True,
             records=None):
        """One chat-completion request. This is a dispatcher, not itself a
        generator, so its return shape depends on `stream`:

        - stream=False: blocks and returns a single tuple
          (content, reasoning, tool_calls, usage).
        - stream=True: returns a generator that yields incremental
          (content, reasoning, finished_tool_calls, usage) tuples; the final
          yield carries the full usage and the assembled tool calls.

        either shape raises ApiError on failure: transient failures (network
        errors, 429/5xx) are retried up to self.retries attempts first, and a
        streaming request is only ever retried before its first byte was read.

        `retry=False` makes a single attempt: a transient failure raises
        ApiError at once, flagged `retryable`, and the caller runs the retry
        loop itself with retry_wait() - how the agentic loop surfaces every
        backoff as an event instead of blocking silently in here.

        `records`, when given, is the caller's cai.request_records
        RequestRecords: every HTTP request this call makes is written down in
        it (what was sent, what came back) for the :requests view.

        `system_prompt`, when given, is a ready-built message dict prepended
        to `messages`.

        `interrupt`, when given, is a threading.Event: the blocking work moves
        to a pump thread and this side polls it, so a set interrupt aborts the
        request within _POLL_TICK even while a recv is blocked mid-request. an
        interrupted call comes back as empty content (streaming: the generator
        just ends early) - the caller holds the Event, so it can tell that
        apart from a real empty answer."""
        url = f"{self.base_url}/chat/completions"
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        headers['Content-Type'] = "application/json"

        data = self._request_data(messages,
                                  model,
                                  system_prompt,
                                  tools,
                                  tool_choice,
                                  reasoning_effort,
                                  temperature)

        # nobody is collecting: write the records into a throwaway list, so
        # the request code below never has to ask whether one was given.
        if records is None:
            records = RequestRecords(limit=1)

        if stream:
            data['stream'] = True
            data['stream_options'] = {"include_usage": True}
            if interrupt is None:
                return self._stream(url, headers, data, records, retry=retry)
            return self._stream_polled(url, headers, data, interrupt, retry, records)
        if interrupt is None:
            return self._complete(url, headers, data, records, retry=retry)
        return self._complete_polled(url, headers, data, interrupt, retry, records)

    def _get_data(self, url):
        """GET url and return its JSON 'data' list, or None on any failure
        (network error, non-200, bad JSON)."""
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        try:
            r = requests.get(url,
                             headers=headers,
                             timeout=self.timeout,
                             verify=self.ssl_verify)
        except requests.RequestException as e:
            log.error(f"[!] request {url} failed: {e}")
            return
        if r.status_code != 200:
            log.error(f"[!] request {url} failed with {r.status_code}")
            return
        try:
            return r.json().get('data', [])
        except ValueError as e:
            log.error(f"[!] request {url} returned invalid JSON: {e}")
            return

    def get_models(self):
        """list the provider's models via the OpenAI-style GET /models endpoint.

        returns a list of records - {'id', plus 'price_in'/'price_out' (USD per
        million tokens) and 'context_length' when known} - or None on failure so
        the caller can fall back to a cache.

        provider shapes handled:
          - OpenRouter: /models carries 'pricing' and 'context_length'.
          - vLLM:       /models carries 'max_model_len'.
          - LiteLLM:    /models is the bare OpenAI spec (ids only); the details
                        live on /model/info, which we merge in when /models gave
                        no pricing or context for any model."""
        data = self._get_data(f"{self.base_url}/models")
        if data is None:
            return

        records = []
        detailed = False
        for model in data:
            if not model.get('id'): continue
            record = {}
            record['id'] = model['id']
            pricing = model.get('pricing') or {}
            price_in = _price_per_mtok(pricing.get('prompt'))
            price_out = _price_per_mtok(pricing.get('completion'))
            if price_in is not None:
                record['price_in'] = price_in
            if price_out is not None:
                record['price_out'] = price_out
            context = model.get('context_length') or model.get('max_model_len')
            if context:
                record['context_length'] = context
            if len(record) > 1:
                detailed = True
            records.append(record)

        # bare /models (LiteLLM): enrich pricing/context from /model/info.
        if not detailed:
            self._merge_model_info(records)
        return records

    def _merge_model_info(self, records):
        """fold LiteLLM's /model/info details into records (matched by id). a
        missing endpoint (not LiteLLM) just leaves records untouched."""
        data = self._get_data(f"{self.base_url}/model/info")
        if not data:
            return
        extra_by_id = {}
        for model in data:
            name = model.get('model_name')
            if not name: continue
            info = model.get('model_info') or {}
            extra = {}
            price_in = _price_per_mtok(info.get('input_cost_per_token'))
            price_out = _price_per_mtok(info.get('output_cost_per_token'))
            if price_in is not None:
                extra['price_in'] = price_in
            if price_out is not None:
                extra['price_out'] = price_out
            context = info.get('max_input_tokens') or info.get('max_tokens')
            if context:
                extra['context_length'] = context
            if extra:
                extra_by_id[name] = extra
        for record in records:
            extra = extra_by_id.get(record['id'])
            if extra:
                record.update(extra)

    def retry_wait(self, error, attempt):
        """the retry policy, for one failed attempt: (seconds to wait before
        the next attempt, the attempt budget), or None when `error` is final -
        not a transient failure, or the budget (self.retries, or
        self.rate_limit_retries for a 429) is spent. `attempt` counts the
        attempts made so far, the failed one included."""
        if not error.retryable:
            return None
        attempts = self.retries
        if error.status == 429:
            attempts = self.rate_limit_retries
        if attempt >= attempts:
            return None
        return _retry_delay(error.status, attempt, error.retry_after), attempts

    def _post_with_retry(self, url, headers, data, stream, records,
                         interrupt=None, retry=True):
        """POST one chat request, retrying the transient failures (network
        errors, 429/5xx) with a doubling backoff - a longer one, over more
        attempts (self.rate_limit_retries), for a 429 - or after the wait the
        server's Retry-After header asks for. returns the 200
        response and its RequestRecord (every attempt gets a record of its
        own in `records`; the caller finishes the returned one once it has
        read the body); raises ApiError once the attempts run out or on a permanent
        failure. a streaming caller gets the response back before any body was
        read, so a retry here never duplicates streamed output. a set
        interrupt cuts the backoff wait short and gives up instead of retrying
        into a run nobody wants anymore. retry=False stops after the first
        attempt, leaving the retry loop to the caller."""
        attempt = 0
        while True:
            attempt += 1
            status = None
            retry_after = None
            record = records.open("POST", url, headers, data)
            try:
                r = requests.post(url,
                                  headers=headers,
                                  json=data,
                                  stream=stream,
                                  timeout=self.timeout,
                                  verify=self.ssl_verify)
            except requests.RequestException as e:
                message = f"request {url} failed: {e}"
                record.failed(e)
            else:
                record.responded(r.status_code, r.headers)
                if r.status_code == 200:
                    return r, record
                status = r.status_code
                retry_after = _asked_wait(r.headers)
                body = r.text
                message = f"request {url} failed with {status}: {body[:300]}"
                record.failed(body)
                r.close()
            error = ApiError(message,
                             status=status,
                             retryable=_retryable(status),
                             retry_after=retry_after)
            wait = None
            if retry:
                wait = self.retry_wait(error, attempt)
            if wait is None:
                raise error
            delay, attempts = wait
            log.warning("api: %s; retrying in %.1fs (attempt %d/%d)",
                        message, delay, attempt, attempts)
            if interrupt is None:
                time.sleep(delay)
            elif interrupt.wait(delay):
                raise error

    def _complete(self, url, headers, data, records, interrupt=None, retry=True):
        """Blocking path: one POST (retried while transient), parse the single
        JSON body, return the (content, reasoning, tool_calls, usage) tuple. an
        unusable body raises ApiError - a failed call must never read as the
        model answering with an empty string. a call interrupted mid-POST
        discards the late response and returns empty content instead."""
        r, record = self._post_with_retry(url, headers, data, False, records,
                                          interrupt=interrupt, retry=retry)
        if interrupt is not None and interrupt.is_set():
            r.close()
            record.aborted()
            return "", "", None, {}
        try:
            result = r.json()
        except ValueError as e:
            raise _unusable(record, f"request {url} returned invalid JSON: {e}")

        choices = result.get("choices", [])
        if len(choices) != 1:
            raise _unusable(record, f"request {url} returned {len(choices)} choices, expected 1")

        message = choices[0].get('message', None)
        if not message:
            raise _unusable(record, f"request {url} returned a choice with no message")

        content = message.get('content', "")
        reasoning = message.get('reasoning') or message.get('reasoning_content') or ""
        tool_calls = message.get('tool_calls', None)
        usage = result.get('usage', {})

        record.finished(content, reasoning, tool_calls, usage)
        return content, reasoning, tool_calls, usage

    def _stream(self, url, headers, data, records, interrupt=None, flight=None,
                retry=True):
        """Streaming path: POST with stream=True (retried while transient,
        before any byte was read), parse the SSE `data:` lines, reassemble
        tool-call argument fragments, and yield deltas as they arrive. The
        final yield carries the assembled tool calls + full usage. a drop
        mid-stream raises ApiError without retrying - a retry would replay
        output the consumer already saw.

        `flight`, when given, receives the live response as soon as the POST
        came back, so the foreground poll loop can abort the socket under a
        blocked read."""
        r, record = self._post_with_retry(url, headers, data, True, records,
                                          interrupt=interrupt, retry=retry)
        if flight is not None:
            flight.response = r
        if interrupt is not None and interrupt.is_set():
            r.close()
            record.aborted()
            return

        try:
            yield from self._read_stream(r, url, record, interrupt)
        finally:
            # a stream the consumer walked away from (an interrupt, an early
            # close) never reached its end; one that did is left untouched.
            record.aborted()

    def _read_stream(self, r, url, record, interrupt):
        """read the SSE body of a streaming response: yield the deltas, and
        write the assembled answer into `record` once the stream ends."""
        finished_tool_calls = None
        tool_calls = {}
        usage = {}
        content_parts = []
        reasoning_parts = []

        with r:
            lines = r.iter_lines()
            while True:
                # each read may hit the connection dropping mid-stream; that
                # surfaces from iter_lines as a requests exception, re-raised
                # as ApiError so the consumer sees one uniform failure type.
                try:
                    line = next(lines)
                except StopIteration:
                    break
                except requests.RequestException as e:
                    # an interrupt shuts the socket under this read (see
                    # _abort_flight); that is a kill, not a broken stream.
                    if interrupt is not None and interrupt.is_set():
                        record.aborted()
                    raise _unusable(record, f"request {url} stream aborted: {e}")
                if not line: continue
                if not line.startswith(b"data: "): continue

                payload = line[len(b"data: "):]
                if payload == b"[DONE]": break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as e:
                    log.error(f"[!] request {url} stream returned invalid JSON: {e}")
                    continue
                record.chunks += 1

                # the final usage chunk has empty choices.
                if chunk.get('usage'):
                    usage = chunk['usage']

                if len(chunk.get("choices") or []) != 1: continue

                choice = chunk["choices"][0]
                finish_reason = choice.get('finish_reason', None)
                delta = choice["delta"]

                # process tool-call fragments before snapshotting finish_reason,
                # so the snapshot captures data from the final chunk too.
                if "tool_calls" in delta:
                    for tool_call in delta['tool_calls']:
                        idx = tool_call['index']
                        if idx not in tool_calls:
                            function = {}
                            function['name'] = tool_call.get('function', {}).get('name')
                            function['arguments'] = ""

                            tool_calls[idx] = {}
                            tool_calls[idx]['index'] = tool_call.get('index')
                            tool_calls[idx]['id'] = tool_call.get('id')
                            tool_calls[idx]['type'] = "function"
                            tool_calls[idx]['function'] = function
                        args = tool_call.get('function', {}).get('arguments')
                        if args:
                            tool_calls[idx]['function']['arguments'] += args

                # only snapshot when tool_calls is non-empty - some providers
                # fire finish_reason="tool_calls" twice, and the second time
                # tool_calls is already reset to {}.
                if finish_reason in ("tool_calls", "tool_use") and tool_calls:
                    finished_tool_calls = list(tool_calls.values())
                    tool_calls = {}

                content = delta.get('content', None)
                reasoning = delta.get('reasoning') or delta.get('reasoning_content')

                if content:
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content or reasoning or tool_calls:
                    yield content, reasoning, finished_tool_calls, {}
            record.finished("".join(content_parts),
                            "".join(reasoning_parts),
                            finished_tool_calls,
                            usage)
            yield None, None, finished_tool_calls, usage

    def _complete_polled(self, url, headers, data, interrupt, retry, records):
        """_complete behind a pump thread: the POST blocks over there while
        this side polls, so a set interrupt returns at once - even while the
        pump is still deep in a socket recv. the pump discards a response that
        lands after the abort."""
        flight = _Flight()
        thread = threading.Thread(target=self._pump_complete,
                                  args=(url, headers, data, interrupt, flight, retry, records),
                                  daemon=True,
                                  name="cai-api-pump")
        thread.start()
        while True:
            if interrupt.is_set():
                _abort_flight(flight)
                return "", "", None, {}
            try:
                kind, value = flight.queue.get(timeout=_POLL_TICK)
            except queue.Empty:
                continue
            if kind == 'error':
                raise value
            return value

    def _pump_complete(self, url, headers, data, interrupt, flight, retry, records):
        try:
            result = self._complete(url, headers, data, records,
                                    interrupt=interrupt, retry=retry)
            flight.queue.put(('done', result))
        except Exception as e:
            flight.queue.put(('error', e))

    def _stream_polled(self, url, headers, data, interrupt, retry, records):
        """_stream behind a pump thread: the pump reads the SSE lines and
        queues the yielded tuples while this generator polls the queue, so a
        set interrupt ends the stream within _POLL_TICK even while the pump is
        blocked waiting for the next chunk. the abort shuts the socket down,
        so the pump dies right away instead of running out the read timeout."""
        flight = _Flight()
        thread = threading.Thread(target=self._pump_stream,
                                  args=(url, headers, data, interrupt, flight, retry, records),
                                  daemon=True,
                                  name="cai-api-pump")
        thread.start()
        try:
            while True:
                if interrupt.is_set(): return
                try:
                    kind, value = flight.queue.get(timeout=_POLL_TICK)
                except queue.Empty:
                    continue
                if kind == 'error':
                    raise value
                if kind == 'done':
                    return
                yield value
        finally:
            # reached on the interrupt return, on an early close by the
            # consumer, and on normal completion (where the pump has already
            # closed the response and the abort is a no-op).
            _abort_flight(flight)

    def _pump_stream(self, url, headers, data, interrupt, flight, retry, records):
        try:
            for item in self._stream(url, headers, data, records, interrupt, flight,
                                     retry=retry):
                flight.queue.put(('item', item))
            flight.queue.put(('done', None))
        except Exception as e:
            flight.queue.put(('error', e))


class SystemOneApi:
    """A minimal System One client - the TypeSafe decision API (Jev): one HTTP
    POST that returns a typed answer per question, never text. the same
    bottom-of-the-stack contract as OpenAiApi - it speaks the wire format,
    retries the transient failures, and raises ApiError for everything it
    cannot recover - but a different protocol, so a different class.

    the endpoint is base_url + /systemone (TypeSafe's own path, which the open
    reproductions copy). OpenRouter serves the identical body and response at
    its Decisions endpoint instead, /api/alpha/decisions on the same host, so
    a base_url on openrouter.ai is rerouted there - the one provider shape
    handled, the way get_models knows OpenRouter's /models."""

    def __init__(self,
                 base_url,
                 api_key,
                 ssl_verify=True,
                 timeout=(10, 120),
                 retries=3,
                 rate_limit_retries=5):
        self.base_url = base_url
        self.api_key = api_key
        self.url = f"{base_url}/systemone"
        parts = urlsplit(base_url)
        if parts.hostname == "openrouter.ai":
            self.url = urlunsplit((parts.scheme, parts.netloc, "/api/alpha/decisions", "", ""))
        self.ssl_verify = ssl_verify
        if not ssl_verify:
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        if isinstance(timeout, (list, tuple)):
            timeout = tuple(timeout)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.rate_limit_retries = max(1, rate_limit_retries)

    def system_one(self, state, questions, model):
        """One System One request: a POST to self.url carrying a `state` (a
        string, or any JSON object/array) and a map of typed questions about it,
        answered all at once. no messages, no streaming, no tools.

        `questions` maps a caller-chosen name to one question dict in the wire
        shape - 'noul' (yes/no), 'choice' (one of named options) or 'score' (an
        ordered scale), each with 'instructions' and 'criteria':

            {"risky": {"type": "noul",
                       "instructions": "Would this tool call damage user data?"},
             "team":  {"type": "choice",
                       "instructions": "Which team should handle this?",
                       "criteria": {"billing": "payments, refunds",
                                    "technical": "bugs, outages"}},
             "anger": {"type": "score",
                       "instructions": "How frustrated is the customer?",
                       "criteria": ["calm", "frustrated", "very angry"]}}

        returns (answers, usage): `answers` is the wire map keyed by the same
        names - {"type": "noul", "noul": 0.92}, {"type": "choice", "choice":
        "technical", "probabilities": {...}, "confidence": 0.82}, {"type":
        "score", "score": 1.6, "legend": {...}, "probabilities": {...},
        "confidence": 0.78} - and `usage` the token counts. transient failures
        (network errors, 429/5xx) are retried up to self.retries attempts
        (self.rate_limit_retries for a 429) with a doubling backoff or the
        server's Retry-After; anything else raises ApiError."""
        url = self.url
        headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        headers['Content-Type'] = "application/json"

        data = {}
        data['model'] = model
        data['state'] = state
        data['questions'] = questions

        attempt = 0
        while True:
            attempt += 1
            status = None
            reply_headers = None
            try:
                r = requests.post(url,
                                  headers=headers,
                                  json=data,
                                  timeout=self.timeout,
                                  verify=self.ssl_verify)
            except requests.RequestException as e:
                error = f"request {url} failed: {e}"
            else:
                if r.status_code == 200:
                    break
                status = r.status_code
                reply_headers = r.headers
                error = f"request {url} failed with {status}: {r.text[:300]}"
                r.close()
            attempts = self.rate_limit_retries if status == 429 else self.retries
            if not _retryable(status) or attempt >= attempts:
                raise ApiError(error, status=status)
            delay = _retry_delay(status, attempt, _asked_wait(reply_headers))
            log.warning("api: %s; retrying in %.1fs (attempt %d/%d)",
                        error, delay, attempt, attempts)
            time.sleep(delay)

        try:
            result = r.json()
        except ValueError as e:
            raise ApiError(f"request {url} returned invalid JSON: {e}")

        answers = result.get('answers')
        if not isinstance(answers, dict):
            raise ApiError(f"request {url} returned no answers")
        usage = result.get('usage', {})
        return answers, usage
