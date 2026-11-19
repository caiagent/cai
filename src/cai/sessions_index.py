"""sessions_index: an in-memory index of the saved sessions, behind :sessions.

The picker repaints on every key and refreshes itself twice a second, so it
cannot afford to touch the disk for either: a .flow is a whole conversation
as JSON, megabytes for a long one. This index reads each file ONCE - again
only when its mtime or size changes - and keeps what the picker asks for:

- the row: its label and the sub-agent sessions it declares (the tree);
- the text: every message flattened to one line, for the search and the
  preview, in the message's own case and lowercased.

Listing, searching and previewing are then pure memory. Reading happens on a
background thread (warm(), started with the TUI), newest session first; a
picker opened before it is done shows the sessions not read yet under their
date alone and fills them in as they land, so it never waits for the disk.

A search pattern is words: a session matches when EVERY word occurs somewhere
in it - any message, or its label - in any order, ignoring case. Subsequence
("fuzzy") matching is deliberately not applied to message bodies: across a
whole conversation nearly any letters occur in order, so it would match
everything. Matches whose label already holds every word rank first, then the
rest; each group stays newest first.

Thread-safe: one lock guards the entries; files are read outside it."""
from __future__ import annotations

import logging
import os
import threading

from cai.session import SessionsRegistry


log = logging.getLogger("cai")

# the flattened text of all indexed sessions may take this much memory
# (counted in characters, both cases). past it, older sessions keep their row
# but not their text: they still list and preview nothing more than a note,
# and match a search on their label only.
TEXT_BUDGET = 256 * 1024 * 1024

# rows of matching messages shown at the top of a filtered session's preview.
_MATCH_ROWS = 8


class _Entry:
    """one indexed session."""

    def __init__(self, path, stamp):
        self.path = path
        self.stamp = stamp           # (mtime_ns, size) the entry was read at
        self.id = os.path.splitext(os.path.basename(path))[0]
        self.label = ""
        self.label_lower = ""
        self.children = []
        self.count = 0               # non-system messages
        self.roles = []              # per message
        self.body = None             # one line per message; None = over budget / unreadable
        self.lower = None            # body, lowercased - what a search scans
        self.failed = False          # the file did not parse

    def text_size(self):
        if self.body is None:
            return 0
        return len(self.body) + len(self.lower)


def _flat(message):
    """a message as one line of text: what the search matches and the preview
    shows. tool calls contribute their names and arguments."""
    from cai.screen.state import _overlay_msg_text

    text = _overlay_msg_text(message)
    if "\n" in text or "\r" in text:
        text = text.replace("\r", " ").replace("\n", " ")
    return text


def _stamp_of(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read(path, stamp, keep_text):
    """build the entry for one .flow file (the slow part: called unlocked)."""
    entry = _Entry(path, stamp)
    try:
        payload = SessionsRegistry.read_flow(path)
    except (OSError, ValueError):
        entry.failed = True
        entry.label = SessionsRegistry.session_stamp(path)
        entry.label_lower = entry.label.lower()
        return entry
    entry.label = SessionsRegistry.session_label(path, payload)
    entry.label_lower = entry.label.lower()
    entry.children = list(payload.get("children") or [])
    lines = []
    for message in (payload.get("messages") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "system": continue
        entry.roles.append(str(message.get("role", "?")))
        if keep_text:
            lines.append(_flat(message))
    entry.count = len(entry.roles)
    if keep_text:
        entry.body = "\n".join(lines)
        entry.lower = entry.body.lower()
    return entry


class SessionsIndex:
    """the index. one per process (see default()); every method may be called
    from any thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = {}           # path -> _Entry
        self._order = []             # paths, newest first, as of the last sync
        self._text = 0               # characters of body+lower held
        self._previews = {}          # (path, stamp, width, query) -> [lines, messages rendered]
        self._lines = {}             # path -> (stamp, body lines, lower lines): the last few previewed
        self._warming = None

    # --- keeping up with the disk -------------------------------------------

    def _pending(self):
        """sync the path list with the disk and return the (path, stamp)
        pairs that still need reading, newest first. cheap: a listdir and a
        stat per file, no file is opened."""
        paths = SessionsRegistry.list_sessions()
        todo = []
        with self._lock:
            live = set(paths)
            for path in list(self._entries):
                if path in live: continue
                self._text -= self._entries.pop(path).text_size()
                self._lines.pop(path, None)
            self._order = paths
            for path in paths:
                stamp = _stamp_of(path)
                if stamp is None: continue
                entry = self._entries.get(path)
                if entry is not None and entry.stamp == stamp: continue
                todo.append((path, stamp))
        return todo

    def _store(self, entry):
        with self._lock:
            old = self._entries.get(entry.path)
            if old is not None:
                self._text -= old.text_size()
            self._entries[entry.path] = entry
            self._text += entry.text_size()
            self._lines.pop(entry.path, None)
            if len(self._previews) > 256:
                self._previews.clear()

    def sync(self, limit=None):
        """read the sessions that are new or changed on disk - all of them,
        or at most `limit` (the picker's refresh reads a couple per tick so a
        key press is never stuck behind a long read). returns how many are
        still unread."""
        todo = self._pending()
        done = 0
        for path, stamp in todo:
            if limit is not None and done >= limit:
                break
            with self._lock:
                keep_text = self._text < TEXT_BUDGET
            self._store(_read(path, stamp, keep_text))
            done += 1
        return len(todo) - done

    def warm(self):
        """start reading every session on a background thread (once at a
        time). the TUI calls this at startup so :sessions opens on a full
        index."""
        with self._lock:
            if self._warming is not None and self._warming.is_alive():
                return
            thread = threading.Thread(target=self._warm, daemon=True, name="cai-sessions-index")
            self._warming = thread
        thread.start()

    def refresh(self, wait=0.0):
        """notice what changed on disk and get it read in the background;
        never blocks on a file. `wait` gives that read up to so many seconds
        to finish first - the picker passes a blink's worth when it opens, so
        an ordinary sessions folder is complete on the first frame."""
        if not self._pending():
            return
        self.warm()
        if wait <= 0:
            return
        with self._lock:
            thread = self._warming
        if thread is not None:
            thread.join(wait)

    def _warm(self):
        try:
            self.sync()
        except Exception:
            log.exception("sessions index: warm-up failed")

    # --- the picker's three questions ----------------------------------------

    def nodes(self):
        """the tree rows: one node per saved session, nested under the session
        that declares it as a child; a declared child with no file of its own
        is a 'dead_end' leaf. a session not read yet shows under its date."""
        with self._lock:
            order = list(self._order)
            entries = dict(self._entries)
        nodes = {}
        declared = []
        for path in order:
            node_id = os.path.splitext(os.path.basename(path))[0]
            entry = entries.get(path)
            node = {}
            node["id"] = node_id
            node["parent"] = None
            node["path"] = path
            node["dead_end"] = False
            if entry is None:
                node["name"] = SessionsRegistry.session_stamp(path) + "  …"
            else:
                node["name"] = entry.label
                for child_id in entry.children:
                    declared.append((node_id, child_id))
            nodes[node_id] = node
        for parent_id, child_id in declared:
            child = nodes.get(child_id)
            if child is None:
                leaf = {}
                leaf["id"] = child_id
                leaf["parent"] = parent_id
                leaf["name"] = child_id
                leaf["path"] = None
                leaf["dead_end"] = True
                nodes[child_id] = leaf
                continue
            child["parent"] = parent_id
        return list(nodes.values())

    def search(self, query):
        """ids of the sessions matching `query`, best first (see the module
        docstring for what matches). an empty query matches nothing - the
        caller shows everything then."""
        terms = query.lower().split()
        if not terms:
            return []
        with self._lock:
            order = list(self._order)
            entries = dict(self._entries)
        in_label = []
        in_text = []
        for path in order:
            entry = entries.get(path)
            if entry is None: continue
            label_hit = True
            hit = True
            for term in terms:
                if term in entry.label_lower: continue
                label_hit = False
                if entry.lower is None or term not in entry.lower:
                    hit = False
                    break
            if not hit: continue
            if label_hit:
                in_label.append(entry.id)
            else:
                in_text.append(entry.id)
        return in_label + in_text

    def _split(self, entry):
        """the body of an entry as per-message lines (original, lowercased).
        split once and kept for the last few sessions previewed."""
        with self._lock:
            held = self._lines.get(entry.path)
            if held is not None and held[0] == entry.stamp:
                return held[1], held[2]
        body = entry.body.split("\n")
        lower = entry.lower.split("\n")
        if entry.count == 0:
            body = []
            lower = []
        with self._lock:
            if len(self._lines) >= 8:
                self._lines.clear()
            self._lines[entry.path] = (entry.stamp, body, lower)
        return body, lower

    def preview(self, path, width, max_lines, query=""):
        """the first `max_lines` lines of a session's preview: a '<n> messages'
        header, then the conversation from its FIRST message, one line each.
        with a query, the messages that match it are listed ahead of the
        conversation, each cut to show its match. rendered lines are cached
        and extended as the pane pages down, so a repaint costs a slice."""
        from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET
        from cai.screen.render import _preview_line

        with self._lock:
            entry = self._entries.get(path)
        if entry is None:
            return [f"{SGR_DIM_GRAY}reading…{SGR_RESET}"]
        if entry.failed:
            return [f"{SGR_DIM_GRAY}could not read session{SGR_RESET}"]
        if entry.body is None:
            return [f"{entry.count} messages"[:width], "",
                    f"{SGR_DIM_GRAY}too many sessions to keep in memory - "
                    f"resume it to read it{SGR_RESET}"]
        key = (path, entry.stamp, width, query)
        with self._lock:
            cached = self._previews.get(key)
        body, lower = self._split(entry)
        if cached is None:
            lines = self._preview_head(entry, body, lower, width, query)
            cached = [lines, 0]
        lines, rendered = cached
        # extend to cover the request: header rows + one row per message.
        while len(lines) < max_lines and rendered < len(body):
            role = {"role": entry.roles[rendered]}
            lines.append(_preview_line(rendered, role, width, text=body[rendered]))
            rendered += 1
        cached[1] = rendered
        with self._lock:
            self._previews[key] = cached
        return lines[:max_lines]

    def _preview_head(self, entry, body, lower, width, query):
        """the rows above the conversation: the count, and - under a query -
        the messages matching it, each excerpted around its first match."""
        from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET
        from cai.screen.render import _preview_line

        terms = query.lower().split()
        if not terms:
            return [f"{entry.count} messages"[:width], ""]
        matches = []
        total = 0
        for i, line in enumerate(lower):
            at = -1
            for term in terms:
                found = line.find(term)
                if found < 0: continue
                if at < 0 or found < at:
                    at = found
            if at < 0: continue
            total += 1
            if len(matches) >= _MATCH_ROWS: continue
            # show the match with a little lead-in. positions come from the
            # lowercased line; lowercasing can shift a few characters in
            # exotic scripts, which only nudges where the excerpt starts.
            start = max(0, at - 24)
            text = body[i][start:]
            if start:
                text = "…" + text
            matches.append(_preview_line(i, {"role": entry.roles[i]}, width, text=text))
        head = [f"{entry.count} messages · {total} matching"[:width], ""]
        if not matches:
            return head
        head.extend(matches)
        if total > len(matches):
            head.append(f"{SGR_DIM_GRAY}… {total - len(matches)} more{SGR_RESET}")
        head.append("")
        return head


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def default():
    """the process's sessions index (created on first use)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = SessionsIndex()
        return _DEFAULT
