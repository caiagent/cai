"""tui.status: what the TUI shows around the conversation - the status line
and the thread that renders it, the pending / sub-agent chips, the Tab status
panel, the poll loop that keeps them fresh, and the agent's recovered errors."""
import threading
import time

from cai import usage
from cai.screen import Screen
from cai.screen.ansi import SGR_RESET
from cai.tui.client import _agent_control


# how often the status thread resamples the signals and repaints the line.
_REFRESH_INTERVAL = 0.1


# a stream is "stalled" once this many seconds pass with no new token; the status
# then falls back from responding/reasoning to waiting.
_STALL_SECONDS = 3.0


# how long a ctx.ui.status(...) note stays on the status line before the normal
# busy/idle readout takes over again.
_NOTE_SECONDS = 4.0


# context-window size used for the ctx % readout when config.json sets no
# 'default_context_size'. matches the reference's fallback.
_DEFAULT_CONTEXT_SIZE = 1_000_000


def _status_text(model, state):
    """the left-hand status string: 'model | state'. the screen adds the
    '-- MODE --' prefix and scroll % itself."""
    parts = []
    if model:
        parts.append(model)
    parts.append(state)
    return " | ".join(parts)


class _Status:
    """holds the live status signals and renders the status line.

    the worker updates the signals as the agent streams (busy/idle, the kind of
    the last delta, and when the last token arrived); the status thread calls
    refresh() on a timer. keeping the decision in refresh() - rather than
    painting on each event - is what lets a quiet stream fall back to 'waiting'
    on the clock alone."""

    def __init__(self, screen, model, registry, fallback_limit):
        self._screen = screen
        self._model = model
        # the context-window limit for the ctx % readout: the current model's
        # cached context_length from the registry when known, else this fallback.
        self._registry = registry
        self._fallback_limit = fallback_limit
        self._lock = threading.Lock()
        self._busy = False
        self._phase = None         # None | "tool" | "waiting"
        self._stream_kind = None   # None | "responding" | "reasoning"
        self._last_char = 0.0      # time.monotonic() of the last streamed token
        self._tokens = 0           # exact tokens from the api's usage channel (0 until a sample)
        self._context_limit = fallback_limit
        self._limit_model = None   # the model _context_limit was resolved for
        self._resolve_limit()
        # the latest real usage sample (tokens measured when the conversation
        # held sample_chars chars), surfaced to the :messages overlay so its
        # per-message token math matches the status line.
        self._sample_tokens = 0
        self._sample_chars = 0
        self._last = None          # last (text, right) painted; unchanged is a no-op
        self._note = ""            # transient ctx.ui.status note shown over the state
        self._note_at = 0.0        # time.monotonic() the note was set

    # --- signal updates, called by the worker thread as it streams a run ---

    def busy(self):
        with self._lock:
            self._busy = True
            self._phase = "waiting"
            self._stream_kind = None
            self._last_char = time.monotonic()

    def idle(self):
        with self._lock:
            self._busy = False

    def stream(self, kind):
        with self._lock:
            self._stream_kind = kind
            self._phase = None
            self._last_char = time.monotonic()

    def tool(self):
        with self._lock:
            self._phase = "tool"

    def tool_done(self):
        # tool finished; the model is called again, so we are waiting on it.
        with self._lock:
            self._phase = "waiting"
            self._stream_kind = None

    def set_tokens(self, tokens):
        # the exact token count from the api's usage channel, pushed by the
        # worker when a real sample arrives; we only format it into the readout.
        with self._lock:
            self._tokens = tokens

    def set_note(self, message):
        # a hook's ctx.ui.status(...) reaches here over the wire; show it on the
        # status line for _NOTE_SECONDS, then let the normal readout resume.
        with self._lock:
            self._note = message or ""
            self._note_at = time.monotonic()

    def limit_for(self, model):
        """the context window of any model - the registry's cached
        context_length, else the config fallback - for a readout about an
        agent that is not this tui's own (the :agents attach view)."""
        limit = None
        if self._registry is not None:
            limit = self._registry.context_length(model)
        if limit:
            return int(limit)
        return self._fallback_limit

    def _resolve_limit(self):
        # cache the current model's context window (cache-only registry read),
        # falling back when it is unknown. caller manages locking.
        limit = None
        if self._registry is not None:
            limit = self._registry.context_length(self._model)
        if limit:
            self._context_limit = int(limit)
        else:
            self._context_limit = self._fallback_limit
        self._limit_model = self._model

    def refresh_limit(self):
        # re-resolve the limit for the current model - called once the catalogue
        # cache has been warmed in the background so a freshly-known
        # context_length is picked up.
        with self._lock:
            self._resolve_limit()

    def set_model(self, model):
        # reflect a model switch (the :models picker) on the status line, and
        # re-resolve its context-window limit.
        with self._lock:
            self._model = model
            self._resolve_limit()

    def set_sample(self, sample_tokens, sample_chars):
        with self._lock:
            self._sample_tokens = sample_tokens
            self._sample_chars = sample_chars

    def ctx_snapshot(self):
        # (context_limit, sample_tokens, sample_chars) for the :messages overlay.
        with self._lock:
            return self._context_limit, self._sample_tokens, self._sample_chars

    # --- render, called by the status thread on a timer ---

    def _state(self, now):
        # one decision ladder over the signals (mirrors the reference's
        # _refresh_status): a stream quiet past _STALL_SECONDS reads as waiting.
        if not self._busy:
            return "idle"
        if self._phase == "tool":
            return "tools"
        if self._stream_kind and now - self._last_char <= _STALL_SECONDS:
            return self._stream_kind
        return "waiting"

    def refresh(self):
        with self._lock:
            if self._limit_model != self._model:
                self._resolve_limit()
            now = time.monotonic()
            if self._note and now - self._note_at < _NOTE_SECONDS:
                text = self._note
            else:
                text = _status_text(self._model, self._state(now))
            right = usage.format_ctx(self._tokens, self._context_limit)
            key = (text, right)
            if key == self._last:
                return
            self._last = key
        # paint outside the lock: set_status takes the screen's own render lock.
        self._screen.set_status(text, right)


class _StatusLoop(threading.Thread):
    """repaints the status line every _REFRESH_INTERVAL until stop is set.

    sampling on a timer (rather than only on events) is what makes a stalled
    stream fall back to 'waiting' - no token arrives to trigger a refresh, so
    this loop is the only thing that notices the silence."""

    def __init__(self, status, stop):
        super().__init__(daemon=True, name="cai-tui-status")
        self._status = status
        self._stop_event = stop

    def run(self):
        while not self._stop_event.wait(_REFRESH_INTERVAL):
            self._status.refresh()


def _pending_chip_lines(user, steer):
    """the pending widget body: an azure pill counting steer texts the agent
    has taken but not yet folded into the run, above a green pill counting user
    turns queued locally behind the running one. each row shows only while its
    count is non-zero; an all-zero pair paints nothing. both pills share one
    width so the block reads as a column."""
    from cai.screen.ansi import SGR_AZURE_ON_DGRAY, SGR_GREEN_ON_DGRAY
    from cai.screen.chip import Chip

    rows = []
    if steer > 0:
        rows.append((f"{steer} steering", SGR_AZURE_ON_DGRAY))
    if user > 0:
        rows.append((f"{user} queued", SGR_GREEN_ON_DGRAY))
    width = 0
    for text, _sgr in rows:
        width = max(width, len(text))
    lines = []
    for text, sgr in rows:
        chip = Chip(text.ljust(width), sgr=sgr)
        lines.extend(chip.lines())
    return lines


class _Pending:
    """the live counts behind the 'pending' hover widget: user turns queued
    locally while a run is in flight, and steer texts the agent has accepted
    but not yet folded into the run. the input loop and worker move the user
    count (client-side, instant); the chips loop polls the agent for the steer
    count. this object alone touches the widget, repainting only on a change so
    an idle session paints nothing."""

    def __init__(self, screen):
        self._screen = screen
        self._lock = threading.Lock()
        self._user = 0
        self._steer = 0

    def counts(self):
        """(user turns queued, steer texts pending) - what the status panel shows."""
        with self._lock:
            return self._user, self._steer

    def user_queued(self):
        with self._lock:
            self._user += 1
        self.refresh()

    def user_started(self):
        with self._lock:
            if self._user > 0:
                self._user -= 1
        self.refresh()

    def user_removed(self):
        """a queued prompt was dropped from :pending before it ran."""
        self.user_started()

    def set_steer(self, count):
        with self._lock:
            if count == self._steer: return
            self._steer = count
        self.refresh()

    def refresh(self):
        """rebuild the widget from the current counts. safe to call from any
        thread; add_widget/remove_widget take the render lock."""
        with self._lock:
            user = self._user
            steer = self._steer
        lines = _pending_chip_lines(user, steer)
        if not lines:
            self._screen.remove_widget("pending")
            return
        self._screen.add_widget("pending", lines, bottom=True)


def _running_subagents(client):
    """display names of the live sub-agents descended from this session's
    agent, read solely over the agents sockets (like the :agents view):
    every live agent answers get_info, and the descendants are walked
    through the children lists starting from this agent."""
    from cai.agents_registry import AgentsRegistry

    infos = {}
    for name in AgentsRegistry.list_names():
        info = _agent_control(name, "get_info")
        if info is None: continue
        infos[name] = info
    self_name = client.get_info().get("name", "")
    self_info = infos.get(self_name) or {}

    running = []
    pending = list(self_info.get("children") or [])
    seen = set()
    while pending:
        child_id = pending.pop(0)
        if child_id in seen: continue
        seen.add(child_id)
        info = infos.get(child_id)
        if info is None: continue
        running.append(info.get("name") or child_id)
        pending.extend(info.get("children") or [])
    return running


def _agent_chip_lines(names):
    """the agents widget body: one yellow pill per running sub-agent, all
    padded to one width so the column reads as a block. no names, no lines."""
    from cai.screen.ansi import SGR_YELLOW_ON_DGRAY
    from cai.screen.chip import Chip

    width = 0
    for name in names:
        width = max(width, len(name))
    lines = []
    for name in names:
        chip = Chip(name.ljust(width), sgr=SGR_YELLOW_ON_DGRAY)
        lines.extend(chip.lines())
    return lines


class _AgentChips:
    """the 'agents' hover widget: a pill per live sub-agent, stacked above the
    prompt beside the pending chips. the poll loop feeds it the running names;
    it repaints only when they change, so an idle session paints nothing."""

    def __init__(self, screen):
        self._screen = screen
        self._lock = threading.Lock()
        self._names = []

    def set_names(self, names):
        names = list(names)
        with self._lock:
            if names == self._names: return
            self._names = names
        if not names:
            self._screen.remove_widget("agents")
            return
        self._screen.add_widget("agents", _agent_chip_lines(names), bottom=True)


# the status panel's geometry: the widest it grows, the narrowest it is still
# worth drawing, and the label column (the longest label plus its count).
_STATUS_MAX_WIDTH = 76


_STATUS_MIN_WIDTH = 24


_STATUS_LABEL_WIDTH = 12


def _clip_tail(text, width):
    """text cut to width columns keeping its end (the informative part of a
    path), the cut marked with a leading ellipsis."""
    from cai.screen.ansi import display_truncate, display_width

    if display_width(text) <= width: return text
    if width <= 1: return "…"[:width]
    tail = display_truncate(text[::-1], width - 1)[::-1]
    return "…" + tail


def _clip_head(text, width):
    """text cut to width columns keeping its start, the cut marked with a
    trailing ellipsis."""
    from cai.screen.ansi import display_truncate, display_width

    if display_width(text) <= width: return text
    if width <= 1: return "…"[:width]
    return display_truncate(text, width - 1) + "…"


def _pack_names(names, width, sgr):
    """names flowed into as few rows of at most width columns as they need,
    separated by a dim dot. each row is a list of (text, sgr) segments."""
    from cai.screen.ansi import SGR_DIM_GRAY, display_width

    rows = []
    row = []
    used = 0
    for name in names:
        name = _clip_head(name, width)
        need = display_width(name)
        if row and used + 3 + need > width:
            rows.append(row)
            row = []
            used = 0
        if row:
            row.append((" · ", SGR_DIM_GRAY))
            used += 3
        row.append((name, sgr))
        used += need
    if row:
        rows.append(row)
    return rows


def _tool_rows(names, processes, width):
    """a group of tools as rows of at most width columns, by where each one
    runs: the function tools (cai's own process) packed first, then one block
    per tool process - an MCP server or the python tool (processes: {name:
    {kind, sandbox, tools}}) - headed by its name, then the sandbox it runs
    behind, then its tools without the server prefix."""
    from cai.screen.ansi import SGR_DIM_GRAY, SGR_GREEN, SGR_YELLOW, SGR_BOLD

    owner = {}
    for process, entry in processes.items():
        for tool in entry.get("tools") or []:
            owner[tool] = process
    plain = []
    grouped = {}
    for name in names:
        process = owner.get(name)
        if process is None:
            plain.append(name)
            continue
        grouped.setdefault(process, []).append(name)

    rows = _pack_names(plain, width, SGR_GREEN)
    inner = max(1, width - 2)
    for process in sorted(grouped.keys()):
        entry = processes[process]
        rows.append([("▸ ", SGR_DIM_GRAY), (_clip_head(process, inner), SGR_BOLD)])
        sandbox = entry.get("sandbox")
        if entry.get("kind") == "remote":
            rows.append([("  ", ""), ("remote, no sandbox", SGR_DIM_GRAY)])
        elif sandbox:
            label = "sandbox "
            command = _clip_head(" ".join(sandbox), max(1, inner - len(label)))
            rows.append([("  ", ""), (label, SGR_DIM_GRAY), (command, "")])
        else:
            rows.append([("  ", ""), ("no sandbox", SGR_YELLOW)])
        short = []
        prefix = process + "__"
        for name in grouped[process]:
            # the python tool is its own process: its name is the heading.
            if name == process: continue
            if name.startswith(prefix):
                name = name[len(prefix):]
            short.append(name)
        for packed in _pack_names(short, inner, SGR_GREEN):
            rows.append([("  ", "")] + packed)
    return rows


def _status_rows(model, cwd, skills, skill_tools, processes, tools, subagents, allowed, disallowed, user, steer, width):
    """the status panel body as rows of (text, sgr) segments, at most width
    columns each: a label column and the values beside it. the model and cwd
    lead (the fs sandbox and relative paths resolve against the cwd); every
    skill is a bold row with its own tools (skill_tools: {skill: [tools]})
    indented beneath it, grouped by the process that serves them - each MCP
    server and the python tool with its sandbox, see _tool_rows - and the
    tools no skill declares get a section of their own, grouped the same
    way; a section with nothing to list is left out, so path grants and the
    pending counts only show once there are any."""
    from cai.screen.ansi import (
        SGR_BOLD, SGR_CYAN, SGR_DIM_GRAY, SGR_GREEN, SGR_RED, SGR_YELLOW,
        SGR_BOLD_AZURE, SGR_MAGENTA,
    )

    value_width = max(1, width - _STATUS_LABEL_WIDTH)
    rows = []

    def section(label, count, value_rows):
        """one labelled block: the label (and a dim count) on its first row,
        the label column left blank on the rows that follow."""
        head = [(label, SGR_CYAN)]
        used = len(label)
        if count is not None:
            tag = f" {count}"
            head.append((tag, SGR_DIM_GRAY))
            used += len(tag)
        head.append((" " * max(1, _STATUS_LABEL_WIDTH - used), ""))
        indent = [(" " * _STATUS_LABEL_WIDTH, "")]
        for i, value_row in enumerate(value_rows):
            if i == 0:
                rows.append(head + value_row)
                continue
            rows.append(indent + value_row)

    if model:
        section("model", None, [[(_clip_head(model, value_width), SGR_BOLD)]])
    if cwd:
        section("cwd", None, [[(_clip_tail(cwd, value_width), "")]])
    if rows:
        rows.append([])

    before = len(rows)
    # each skill heads the tools it brought in (only the ones still selected);
    # tools no skill declares - picked by hand in :tools - follow on their own.
    selected = set(tools)
    claimed = set()
    for skill in skills:
        own = []
        for tool in skill_tools.get(skill) or []:
            if tool not in selected: continue
            own.append(tool)
            claimed.add(tool)
        value_rows = [[("◆ ", SGR_MAGENTA), (_clip_head(skill, value_width - 2), SGR_BOLD)]]
        for tool_row in _tool_rows(own, processes, value_width - 2):
            value_rows.append([("  ", "")] + tool_row)
        label = ""
        count = None
        if skill == skills[0]:
            label = "skills"
            count = len(skills)
        section(label, count, value_rows)
    loose = []
    for tool in tools:
        if tool in claimed: continue
        loose.append(tool)
    if loose:
        section("tools", len(loose), _tool_rows(loose, processes, value_width))
    if subagents:
        agent_rows = []
        for name in subagents:
            agent_rows.append([("● ", SGR_YELLOW), (_clip_head(name, value_width - 2), "")])
        section("agents", len(subagents), agent_rows)
    if allowed:
        path_rows = []
        for path in allowed:
            path_rows.append([("+ ", SGR_GREEN), (_clip_tail(path, value_width - 2), "")])
        section("allowed", None, path_rows)
    if disallowed:
        path_rows = []
        for path in disallowed:
            path_rows.append([("- ", SGR_RED), (_clip_tail(path, value_width - 2), "")])
        section("denied", None, path_rows)
    if steer > 0 or user > 0:
        counts = []
        if steer > 0:
            counts.append((f"{steer} steering", SGR_BOLD_AZURE))
        if steer > 0 and user > 0:
            counts.append((" · ", SGR_DIM_GRAY))
        if user > 0:
            counts.append((f"{user} queued", SGR_GREEN))
        section("pending", None, [counts])
    if len(rows) == before:
        rows.append([("no skills or tools active", SGR_DIM_GRAY)])
    return rows


def _status_panel_lines(rows, max_rows, max_cols):
    """the rows from _status_rows inside a rounded frame titled 'status',
    every line the same width so the block reads as a window. the frame is
    as wide as its longest row (capped by the caller's row width); a body
    taller than max_rows is cut and closed with a '+N more' row. returns []
    when the viewport is too small to hold a frame at all."""
    from cai.screen.ansi import SGR_BOLD_AZURE, SGR_DIM_GRAY, SGR_RESET, display_width

    title = " status "
    hint = " any key to close "
    inner = len(title) + len(hint) + 2
    for row in rows:
        width = 0
        for text, _sgr in row:
            width += display_width(text)
        inner = max(inner, width)
    # one column of border and two of padding on each side.
    if inner + 6 > max_cols or max_rows < 5: return []

    body_room = max_rows - 4
    shown = rows
    if len(rows) > body_room:
        shown = rows[:body_room - 1]
        shown.append([(f"+{len(rows) - len(shown)} more", SGR_DIM_GRAY)])

    frame = SGR_DIM_GRAY
    span = inner + 4
    lines = []
    lines.append(f"{frame}╭─{SGR_RESET}{SGR_BOLD_AZURE}{title}{SGR_RESET}"
                 f"{frame}{'─' * (span - 1 - len(title))}╮{SGR_RESET}")
    blank = f"{frame}│{SGR_RESET}{' ' * span}{frame}│{SGR_RESET}"
    lines.append(blank)
    for row in shown:
        cells = []
        width = 0
        for text, sgr in row:
            width += display_width(text)
            if sgr:
                cells.append(f"{sgr}{text}{SGR_RESET}")
                continue
            cells.append(text)
        pad = " " * (inner - width)
        lines.append(f"{frame}│{SGR_RESET}  {''.join(cells)}{pad}  {frame}│{SGR_RESET}")
    lines.append(blank)
    lines.append(f"{frame}╰{'─' * (span - 1 - len(hint))}{hint}─╯{SGR_RESET}")
    return lines


class _StatusPanel:
    """the Tab-toggled status panel: a framed window in the middle of the
    viewport showing the agent's model and cwd, active skills, the MCP servers
    behind their tools (each with its sandbox) and the tools, path
    grants/denies (read over the wire), the live sub-agents (fed by the
    poll loop), and the pending counts. it hovers - the conversation and the
    prompt stay live beneath it. toggle() shows/hides it and paints at once;
    once shown, the next keypress hides it (the screen calls hide(): Tab and
    Esc only close it, any other key closes it and still does its job);
    while visible the poll loop refreshes it every tick, so :skills / :tools /
    a model switch / a sub-agent starting / a terminal resize show up without
    any call-site wiring. hidden is the start state."""

    def __init__(self, screen, client, pending):
        self._screen = screen
        self._client = client
        self._pending = pending
        self._lock = threading.Lock()
        self._visible = False
        self._subagents = []

    @property
    def visible(self):
        with self._lock:
            return self._visible

    def toggle(self):
        with self._lock:
            self._visible = not self._visible
            visible = self._visible
        if not visible:
            self._screen.remove_widget("status")
            return
        self.refresh()

    def hide(self):
        with self._lock:
            self._visible = False
        self._screen.remove_widget("status")

    def set_subagents(self, names):
        with self._lock:
            self._subagents = list(names)

    def refresh(self):
        """repaint from live state; a no-op while hidden. wire failures leave
        the last painting in place rather than tearing the panel down."""
        if not self.visible: return
        try:
            info = self._client.get_info()
            skills = list(self._client.get_selected_skills())
            skill_tools = self._client.get_skill_tools()
            processes = self._client.get_tool_processes()
            tools = sorted(self._client.get_selected_tools())
            paths = self._client.get_paths()
        except OSError:
            return
        with self._lock:
            subagents = list(self._subagents)
        user, steer = self._pending.counts()
        max_rows, max_cols = self._screen.viewport_size()
        width = min(_STATUS_MAX_WIDTH, max_cols - 6)
        lines = []
        if width >= _STATUS_MIN_WIDTH:
            rows = _status_rows(info.get("model", ""),
                                info.get("cwd", ""),
                                skills,
                                skill_tools,
                                processes,
                                tools,
                                subagents,
                                paths.get("allowed") or [],
                                paths.get("disallowed") or [],
                                user,
                                steer,
                                width)
            lines = _status_panel_lines(rows, max_rows, max_cols)
        if not lines:
            self._screen.remove_widget("status")
            return
        # hidden while the state was being read (a keypress got in first):
        # painting now would bring back a panel nobody asked for.
        with self._lock:
            if not self._visible: return
            self._screen.add_widget("status", lines, center=True, dismiss=self.hide)


# how often the poll loop re-reads the agent (and the agents sockets) for state
# no event announces: the steer count and the live sub-agents.
_AGENTS_REFRESH = 1.0


class _PollLoop(threading.Thread):
    """polls the agent-side state no event on this run's stream announces,
    every _AGENTS_REFRESH seconds: the steer count into the 'pending' chip,
    and the live sub-agents (walked over the agents sockets) into the
    'agents' chips and the status panel, which is repainted while visible,
    and the agent's unreported errors (a tool that failed to load on a
    session load or a selection applied between turns) onto the screen."""

    def __init__(self, screen, client, stop, pending, agent_chips, status_panel, tabs=None):
        super().__init__(daemon=True, name="cai-tui-poll")
        self._tabs = tabs
        self._screen = screen
        self._client = client
        self._stop_event = stop
        self._pending = pending
        self._agent_chips = agent_chips
        self._status_panel = status_panel

    def run(self):
        while not self._stop_event.wait(_AGENTS_REFRESH):
            if self._tabs is not None:
                self._tabs.refresh_bar()
            try:
                info = self._client.get_info()
                # the error queue is shared by every tab's agent (it lives on
                # the env): only the tab in front drains it, so an error is
                # never written into a conversation nobody is looking at.
                if self._screen.current_focus() == 'main':
                    _show_agent_errors(self._screen, self._client)
            except OSError:
                continue
            self._pending.set_steer(info.get("pending_steer", 0))
            # an agent that never launched a child has none to look for: skip
            # the walk over every live agent's socket.
            names = []
            if info.get("children"):
                try:
                    names = _running_subagents(self._client)
                except OSError:
                    continue
            self._agent_chips.set_names(names)
            self._status_panel.set_subagents(names)
            self._status_panel.refresh()


def _show_agent_errors(screen, client):
    """write the errors the agent recovered from and has not reported yet
    to the conversation, one error line each - e.g. a tool that failed to
    load is skipped on the agent side, so without this it just silently
    isn't there."""
    for error in client.drain_errors():
        screen.write(f"[error: {error}]\n", kind=Screen.ERROR, block=True)
