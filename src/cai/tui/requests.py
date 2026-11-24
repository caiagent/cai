"""tui.requests: the TUI's memory of the HTTP requests its agents sent.

The agentic loop keeps no record of the requests it makes; it only announces
each one to hooks once it ended (after_request, with the request's
RequestRecord in ctx.request). RequestStore is the TUI's hook for that: it keeps the most
recent records, in memory only, and the :requests view reads them back. So
what is kept, and how much, is the TUI's business alone - an SDK or headless
run registers no such hook and keeps nothing.

One store serves the whole TUI: every tab's agent, and every sub-agent they
launch (a child inherits its parent's hooks), reports here, each record filed
under the name of the agent that sent it."""
import threading

from collections import deque


class RequestStore:
    """the most recent requests of all the TUI's agents, oldest first. hooks
    add to it from the agents' worker threads while the main thread reads it
    for the overlay, so the list is guarded by a lock."""

    def __init__(self, settings):
        # the live cai.settings: keep_requests is read on every add, so a
        # change takes effect at once.
        self._settings = settings
        self._kept = deque()        # (agent name, RequestRecord)
        self._lock = threading.Lock()
        self._next_id = 1

    def on_request(self, ctx):
        """the after_request hook: keep the record of a request that just
        ended - answered, failed or interrupted."""
        limit = self._settings.keep_requests
        if limit <= 0:
            return
        agent = ctx.data["agent"]
        with self._lock:
            ctx.request.id = self._next_id
            self._next_id += 1
            self._kept.append((agent.name, ctx.request))
            while len(self._kept) > limit:
                self._kept.popleft()

    def _records_of(self, agent_name):
        with self._lock:
            kept = list(self._kept)
        records = []
        for name, record in kept:
            if name != agent_name: continue
            records.append(record)
        return records

    def summaries(self, agent_name):
        """a summary of each remembered request of one agent, oldest first."""
        summaries = []
        for record in self._records_of(agent_name):
            summaries.append(record.summary())
        return summaries

    def detail(self, agent_name, id):
        """everything about one request of that agent, or None once it is no
        longer remembered."""
        for record in self._records_of(agent_name):
            if record.id != id: continue
            return record.detail()
        return None
