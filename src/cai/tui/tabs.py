"""tui.tabs: vim-like tabs - one served agent, Screen and set of threads per
tab over a shared Environment - the tab line, cloning a session into a new tab
(Ctrl-W T, :checkout). cai.commands.Tabs is the face of these a `:`-command
gets (ctx.tabs); :merge, a bundled extension, is written against it."""
import collections
import copy
import logging
import threading

from cai.origin import origin_of
from cai.screen import Screen
from cai.tui.transcript import _replay_messages
from cai.tui.client import AgentClient, _Jobs, _Worker
from cai.tui.status import (
    _AgentChips,
    _Pending,
    _PollLoop,
    _Status,
    _StatusLoop,
    _StatusPanel,
    _clip_head,
)
from cai.tui.commands import _Command


log = logging.getLogger("cai")


def _clone_point(messages):
    """the conversation a tab cloned mid-run starts from: a snapshot cut back
    to before an assistant message whose tool calls have not all been
    answered yet. the tools are still running in the original tab; the copy
    must not carry half a tool exchange (the api refuses one) nor run those
    calls a second time."""
    messages = copy.deepcopy(list(messages))
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.get("role") == "tool": continue
        if message.get("role") != "assistant": break
        calls = message.get("tool_calls") or []
        if len(calls) > len(messages) - 1 - i:
            return messages[:i]
        break
    return messages


# the tab line's colours, after vim's: the current tab stands out on the
# terminal's own background, the others and the unused width sit on a bar.
_TAB_CURRENT = '\033[1;38;5;45m'


_TAB_OTHER = '\033[38;5;250;48;5;238m'


_TAB_FILL = '\033[48;5;238m'


# the longest label a tab shows before it is cut.
_TAB_LABEL_WIDTH = 24


def _tab_label(messages, fallback):
    """what a tab is called on the tab line: the start of the first prompt
    the user typed in its conversation, or fallback (the session name) for
    a conversation that has none yet."""
    for message in messages:
        if message.get("role") != "user": continue
        if origin_of(message) is not None: continue
        content = message.get("content")
        if not isinstance(content, str): continue
        text = " ".join(content.split())
        if not text: continue
        return _clip_head(text, _TAB_LABEL_WIDTH)
    return fallback


def _tab_line_cells(labels, states, current):
    """the tab line as (text, sgr) segments, one per tab: ' N label ' with
    the current tab highlighted. states is one marker per tab - '' idle,
    '…' running, '?' waiting for an answer, '+' finished since it was last
    looked at - shown after the number, where vim shows its '+'."""
    cells = []
    for i, label in enumerate(labels):
        sgr = _TAB_OTHER
        if i == current:
            sgr = _TAB_CURRENT
        cells.append((f" {i + 1}{states[i]} {label} ", sgr))
    return cells


class _Tab:
    """one tab: an agent of its own, served on its own socket, with its own
    Screen (conversation, input draft, scroll position) and the threads that
    feed it. tabs share nothing but the terminal, the env and the models
    registry, so a tab in the background keeps running and its screen keeps
    filling - it just is not painted until it comes to the front."""

    def __init__(self, agent, shared, tabs):
        from cai.wired_agent import UnixWiredAgent

        self.shared = shared
        self.server = UnixWiredAgent(agent)
        threading.Thread(target=self.server.serve, daemon=True, name="cai-tui-serve").start()
        self.client = AgentClient(self.server.path)
        try:
            self.screen = Screen()
        except OSError:
            self.server.close()
            self.client.close()
            raise
        self.screen.set_palette_commands(shared.palette, shared.arg_commands)
        self.screen.set_cmd_completions(shared.completions)
        self.jobs = _Jobs()
        self.stop = threading.Event()
        self.model = self.client.get_info().get("model", "")
        self.status = _Status(self.screen, self.model, shared.registry, shared.fallback_limit)
        self.status.refresh()   # initial idle paint, before the status thread takes over
        self.screen.set_interrupt_handler(self._on_interrupt)
        self.screen.set_recall_handler(self._on_recall)
        self.pending = _Pending(self.screen)
        self.worker = _Worker(self.client, self.screen, self.jobs, self.stop, self.status,
                              shared.env.settings, self.pending)
        self.worker.start()
        self.status_loop = _StatusLoop(self.status, self.stop)
        self.status_loop.start()
        self.status_panel = _StatusPanel(self.screen, self.client, self.pending)
        self.poll_loop = _PollLoop(self.screen, self.client, self.stop, self.pending,
                                   _AgentChips(self.screen), self.status_panel, tabs)
        # the tab bar's markers: a run seen in flight on the last tick, and a
        # run that finished while the tab was in the background.
        self.was_busy = False
        self.unseen = False
        # work a background thread hands to the main thread (see call_soon).
        self.todo = collections.deque()

    def call_soon(self, fn):
        """run fn on the main thread the next time this tab is in front: a
        blocked prompt() is woken for it, and one not yet entered returns at
        once. how a background thread gets at the terminal (the editor). a
        tab closed first never runs it."""
        self.todo.append(fn)
        self.screen.abort_prompt()

    def _on_interrupt(self):
        # Ctrl-C on an empty prompt: cancel an in-flight run if there is one,
        # and tell the screen we consumed the interrupt so it skips its quit
        # double-tap. otherwise let the default handling run.
        if self.screen.busy:
            self.worker.mark_interrupted()
            self.client.interrupt()
            return True
        return False

    def _on_recall(self):
        # Ctrl-X: take back the last user prompt (e.g. one just Ctrl-C'd before
        # the model answered). only when idle and the conversation actually ends
        # on a plain user message - an interrupted turn that left a partial
        # assistant reply, or a mid-run conversation, is left untouched. returns
        # the text for the screen to drop into the input box; None does nothing.
        if self.screen.busy:
            return None
        msgs = self.client.get_messages()
        if not msgs:
            return None
        last = msgs[-1]
        if last.get("role") != "user":
            return None
        if origin_of(last) is not None:
            return None     # a sub-agent's report, not a prompt the user typed
        content = last.get("content")
        if not isinstance(content, str):
            return None
        self.client.set_messages(msgs[:-1])
        return content

    def marker(self):
        """the tab bar marker for this tab's state."""
        if self.screen.awaiting_answer:
            return "?"
        if self.screen.busy:
            return "…"
        if self.unseen:
            return "+"
        return ""

    def close(self, restore_terminal):
        """stop the tab's threads and connections and close its screen. the
        agent is closed by the caller (_Tabs), which first sees to a scratch
        directory another tab still borrows."""
        self.stop.set()
        self.jobs.put(None)
        self.server.close()
        self.worker.join(timeout=2)
        self.status_loop.join(timeout=2)
        self.poll_loop.join(timeout=2)
        self.client.close()
        self.screen.close(restore_terminal=restore_terminal)


class _Tabs:
    """the open tabs and which one is in front. every switch happens on the
    main thread, between two prompt() calls; the poll loops only read (the
    tab bar), under the lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items = []
        self._index = 0
        self._painted = None   # (screen, cells) of the last tab line painted

    @property
    def current(self):
        return self._items[self._index]

    def count(self):
        return len(self._items)

    def at(self, index):
        """the tab at index (0-based), or None."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def all(self):
        """the open tabs in tab line order (a copy)."""
        with self._lock:
            return list(self._items)

    def number(self, tab):
        """tab's 1-based number on the tab line, or 0 once it is closed."""
        if tab in self._items:
            return self._items.index(tab) + 1
        return 0

    def add(self, tab):
        """put tab right after the current one and bring it to the front. its
        Screen was just created, so it already owns the terminal."""
        with self._lock:
            if self._items:
                self._index += 1
            self._items.insert(self._index, tab)
        self.refresh_bar()

    def leave(self):
        """send the current tab to the background, ahead of a switch or of a
        new tab's Screen taking the terminal."""
        tab = self.current
        tab.status_panel.hide()
        tab.was_busy = tab.screen.busy
        tab.screen.background()

    def switch(self, delta):
        """go delta tabs right (wrapping), like gt / gT."""
        if len(self._items) < 2: return
        self.leave()
        with self._lock:
            self._index = (self._index + delta) % len(self._items)
        self._arrive()

    def _arrive(self):
        tab = self.current
        tab.unseen = False
        self.refresh_bar()
        tab.screen.foreground()

    def close_current(self):
        """close the tab in front and land on its neighbour. the last tab is
        closed too (the terminal is restored) - the caller is quitting."""
        tab = self.current
        last = len(self._items) == 1
        tab.close(restore_terminal=last)
        with self._lock:
            self._items.remove(tab)
            if self._index >= len(self._items):
                self._index = max(0, len(self._items) - 1)
        self._close_agent(tab)
        if last: return
        self._arrive()

    def close(self, tab):
        """close a tab in the background (ctx.tabs.close - :merge); the tab in
        front stays in front and keeps the terminal. the tab in front goes
        through close_current."""
        front = self.current
        if tab is front:
            self.close_current()
            return
        tab.close(restore_terminal=False)
        with self._lock:
            self._items.remove(tab)
            self._index = self._items.index(front)
        self._close_agent(tab)
        self.refresh_bar()

    def close_all(self):
        """quit: close the background tabs, then the one in front, which
        gives the terminal back."""
        front = self.current
        for tab in list(self._items):
            if tab is front: continue
            tab.close(restore_terminal=False)
        front.close(restore_terminal=True)
        tabs = self._items
        with self._lock:
            self._items = []
        for tab in tabs:
            tab.server.agent.close()

    def _close_agent(self, tab):
        """close a closed tab's agent. a tab cloned from it borrows its
        scratch directory - the cloned conversation can name files in there -
        so a scratch this agent owns is handed to a borrower instead of being
        deleted with it."""
        # through the server, not a saved reference: a clone op swaps the
        # served agent, and the current one owns the live scratch.
        agent = tab.server.agent
        if agent._scratch_owned:
            for other in self._items:
                heir = other.server.agent
                if heir._scratch != agent._scratch: continue
                agent._scratch_owned = False
                heir._scratch_owned = True
                break
        agent.close()

    def refresh_bar(self):
        """repaint the tab line on the tab in front, when it changed. like
        vim's, it takes the top row only while there is more than one tab;
        with one tab the screen is as it always was. also where a background
        tab's finished run is noticed (the '+' marker)."""
        with self._lock:
            if not self._items: return
            states = []
            for i, tab in enumerate(self._items):
                if i != self._index:
                    busy = tab.screen.busy
                    if tab.was_busy and not busy:
                        tab.unseen = True
                    tab.was_busy = busy
                states.append(tab.marker())
            screen = self._items[self._index].screen
            cells = None
            if len(self._items) > 1:
                labels = []
                for tab in self._items:
                    # read in-process (no wire round trip per tab per tick);
                    # list() is an atomic snapshot of a list a run may be
                    # appending to.
                    agent = tab.server.agent
                    labels.append(_tab_label(list(agent.messages), agent.name))
                cells = _tab_line_cells(labels, states, self._index)
            if self._painted == (screen, cells): return
            self._painted = (screen, cells)
            screen.set_tabline(cells, _TAB_FILL)


class _Shared:
    """what every tab is built from: the env, the models registry, the
    command palette, and the store the :requests view reads."""

    def __init__(self, env, registry, fallback_limit, palette, arg_commands, completions,
                 requests):
        self.env = env
        self.registry = registry
        self.fallback_limit = fallback_limit
        self.palette = palette
        self.arg_commands = arg_commands
        self.completions = completions
        self.requests = requests


def _open_clone_tab(tabs):
    """Ctrl-W T / :checkout: clone the current tab's agent - conversation,
    model, skills, tools - into a new tab and bring it to the front. the
    clone is its own session (its own name and autosave file) from here on.
    a tab cloned mid-run gets the conversation as of the last finished step;
    the run carries on in the original tab."""
    source = tabs.current
    agent = source.server.agent
    overrides = None
    if source.screen.busy:
        overrides = {}
        overrides["messages"] = _clone_point(agent.get_messages())
    branch = agent.clone(overrides=overrides)
    tabs.leave()
    try:
        tab = _Tab(branch, source.shared, tabs)
    except Exception as e:
        log.exception("opening a tab failed")
        branch.close()
        source.screen.foreground()
        source.screen.write(f"[error: could not open a tab: {type(e).__name__}: {e}]\n",
                            kind=Screen.ERROR, block=True)
        return
    tabs.add(tab)
    _replay_messages(tab.screen, tab.client.get_messages(), source.shared.env.settings)
    tab.status.set_tokens(branch.tokens)
    tab.screen.write(f"[tab {tabs.count()} — cloned from {agent.name}]\n",
                     kind=Screen.META, block=True)
    tab.poll_loop.start()


def _close_tab(ctx):
    """:tabclose - and :q / :quit, which leave the TUI from the last tab."""
    if ctx.tabs.count() > 1:
        ctx.tabs.close_current()
        return False
    if ctx.head == "tabclose":
        ctx.note("the last tab — :q to quit")
        return False
    return True


# the tab commands, in palette order after the session's (cai.tui.commands).
_TAB_COMMANDS = [
    _Command("checkout", "branch this session into a new tab (Ctrl-W T)",
             lambda ctx: _open_clone_tab(ctx.tabs)),
    _Command("tabnext", "go to the next tab (gt)", lambda ctx: ctx.tabs.switch(1)),
    _Command("tabprev", "go to the previous tab (gT)", lambda ctx: ctx.tabs.switch(-1)),
    _Command("tabclose", "close this tab", _close_tab),
    _Command("quit", "close this tab; exit when it is the last one", _close_tab, aliases=("q",)),
    _Command("qa", "exit, closing every tab", lambda ctx: True, aliases=("qall",)),
]
