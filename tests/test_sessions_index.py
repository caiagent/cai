"""Tests for the sessions index behind :sessions (cai.sessions_index): each
.flow is read once and re-read only when it changes, the search covers every
message of every session, and the preview reads from the first message."""
import os
import pty
import shutil
import threading
import time

import pytest

from cai.screen.ansi import ansi_strip
from cai.screen.overlays.model import _draw_model_overlay
from cai.screen.overlays.tree import prompt_tree_overlay
from cai.session import SessionsRegistry
from cai.sessions_index import SessionsIndex


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


def _save(name, messages, children=None, age=0):
    """write a session file; `age` seconds back-dates it (newest sorts first)."""
    payload = SessionsRegistry.flow_payload(messages, "sys prompt", None, [], [], "m",
                                            children=children)
    path = SessionsRegistry.session_path(name)
    SessionsRegistry.write_flow(path, payload)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def _turns(*texts):
    messages = []
    for i, text in enumerate(texts):
        role = "user"
        if i % 2:
            role = "assistant"
        messages.append({"role": role, "content": text})
    return messages


def _index():
    index = SessionsIndex()
    assert index.sync() == 0
    return index


# --- rows -------------------------------------------------------------------

def test_nodes_list_sessions_newest_first_with_their_labels():
    _save("old", _turns("first question", "answer"), age=100)
    _save("new", _turns("second question", "answer"), age=1)
    nodes = _index().nodes()
    assert [n["id"] for n in nodes] == ["new", "old"]
    assert "second question" in nodes[0]["name"]
    assert "(2 msgs)" in nodes[0]["name"]


def test_nodes_nest_children_and_keep_a_missing_child_as_a_dead_end():
    _save("parent", _turns("go"), children=["kid", "gone"], age=5)
    _save("kid", _turns("sub task"), age=4)
    by_id = {}
    for node in _index().nodes():
        by_id[node["id"]] = node
    assert by_id["kid"]["parent"] == "parent"
    assert by_id["gone"]["dead_end"] is True
    assert by_id["gone"]["path"] is None


def test_a_session_not_read_yet_still_has_a_row():
    _save("a", _turns("hello"))
    index = SessionsIndex()
    index._pending()                      # the folder is listed, nothing is read
    nodes = index.nodes()
    assert [n["id"] for n in nodes] == ["a"]
    assert nodes[0]["name"].endswith("…")
    assert "reading" in ansi_strip(index.preview(nodes[0]["path"], 60, 5)[0])


# --- reading each file once -----------------------------------------------------

def test_an_unchanged_file_is_never_read_again(monkeypatch):
    _save("a", _turns("hello"))
    index = _index()
    reads = []
    real = SessionsRegistry.read_flow
    def counting(path):
        reads.append(path)
        return real(path)
    monkeypatch.setattr(SessionsRegistry, "read_flow", staticmethod(counting))
    for _ in range(5):
        assert index.sync() == 0
        index.nodes()
        index.search("hello")
        index.preview(SessionsRegistry.session_path("a"), 60, 10)
    assert reads == []


def test_a_changed_file_is_read_again_and_a_deleted_one_is_dropped():
    path = _save("a", _turns("alpha"), age=10)
    _save("b", _turns("beta"), age=20)
    index = _index()
    assert index.search("gamma") == []
    _save("a", _turns("alpha", "now with gamma"))
    index.sync()
    assert index.search("gamma") == ["a"]
    os.remove(path)
    index.sync()
    assert [n["id"] for n in index.nodes()] == ["b"]
    assert index.search("alpha") == []


def test_sync_limit_reads_a_few_and_reports_the_rest():
    for i in range(4):
        _save(f"s{i}", _turns("x"), age=i)
    index = SessionsIndex()
    assert index.sync(limit=1) == 3
    assert index.sync() == 0


def test_warm_reads_everything_in_the_background():
    _save("a", _turns("needle"))
    index = SessionsIndex()
    index.refresh(wait=5)
    assert index.search("needle") == ["a"]


def test_an_unparsable_file_is_listed_but_not_searched():
    path = SessionsRegistry.session_path("broken")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{not json")
    index = _index()
    assert [n["id"] for n in index.nodes()] == ["broken"]
    assert "could not read" in ansi_strip(index.preview(path, 60, 5)[0])


# --- search -------------------------------------------------------------------

def test_search_reaches_a_message_deep_inside_a_session():
    texts = []
    for i in range(40):
        texts.append(f"filler line {i}")
    texts[37] = "the Zebra crossing was here"
    _save("deep", _turns(*texts), age=2)
    _save("other", _turns("nothing relevant"), age=1)
    index = _index()
    assert index.search("zebra") == ["deep"]
    assert index.search("ZEBRA crossing") == ["deep"]


def test_search_needs_every_word_in_any_order_across_messages():
    _save("both", _turns("apples are red", "bananas are yellow"), age=2)
    _save("one", _turns("apples only"), age=1)
    index = _index()
    assert index.search("bananas apples") == ["both"]
    assert index.search("apples") == ["one", "both"]      # newest first
    assert index.search("apples cherries") == []
    assert index.search("   ") == []


def test_search_covers_tool_calls_and_tool_results():
    call = {"id": "c1", "type": "function",
            "function": {"name": "fs__read_file", "arguments": '{"path": "/etc/hosts"}'}}
    messages = [{"role": "user", "content": "look"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "c1", "content": "127.0.0.1 localhost"}]
    _save("tools", messages)
    index = _index()
    assert index.search("fs__read_file") == ["tools"]
    assert index.search("/etc/hosts") == ["tools"]
    assert index.search("localhost") == ["tools"]


def test_a_label_match_ranks_ahead_of_a_body_match():
    _save("body", _turns("something else", "we discussed kernel modules"), age=1)
    _save("label", _turns("kernel question", "ok"), age=50)
    assert _index().search("kernel") == ["label", "body"]


def test_the_system_prompt_is_not_searched():
    _save("a", _turns("hello"))
    assert _index().search("sys prompt") == []


# --- preview --------------------------------------------------------------------

def test_preview_starts_at_the_first_message():
    texts = []
    for i in range(30):
        texts.append(f"message number {i}")
    path = _save("long", _turns(*texts))
    lines = []
    for line in _index().preview(path, 60, 6):
        lines.append(ansi_strip(line))
    assert lines[0] == "30 messages"
    assert lines[1] == ""
    assert lines[2].startswith("#0 user") and "message number 0" in lines[2]
    assert lines[5].startswith("#3 assistant") and "message number 3" in lines[5]
    assert len(lines) == 6


def test_preview_extends_as_the_pane_pages_down():
    texts = []
    for i in range(30):
        texts.append(f"m{i}")
    path = _save("long", _turns(*texts))
    index = _index()
    assert len(index.preview(path, 60, 5)) == 5
    more = index.preview(path, 60, 100)
    assert len(more) == 32                         # header + blank + every message
    assert ansi_strip(more[-1]).startswith("#29 ")
    assert index.preview(path, 60, 5) == more[:5]    # the cache serves the head again


def test_preview_flattens_a_multiline_message_to_one_row():
    path = _save("ml", _turns("line one\nline two\n\tindented"))
    row = ansi_strip(_index().preview(path, 80, 5)[2])
    assert "line one line two" in row
    assert "\n" not in row and "\t" not in row


def test_preview_lists_the_matching_messages_first_under_a_query():
    texts = []
    for i in range(20):
        texts.append(f"filler {i}")
    texts[15] = "a long preamble before the interesting Needle shows up in the text"
    path = _save("s", _turns(*texts))
    lines = []
    for line in _index().preview(path, 70, 8, query="needle"):
        lines.append(ansi_strip(line))
    assert lines[0] == "20 messages · 1 matching"
    assert lines[2].startswith("#15 assistant")
    assert "Needle" in lines[2]                    # excerpted around the match
    assert lines[3] == ""
    assert lines[4].startswith("#0 user")          # then the session from its start


# --- the overlay ------------------------------------------------------------------

def test_preview_from_top_shows_the_paged_window(capsys):
    def preview(_label, width, max_lines):
        lines = []
        for i in range(min(max_lines, 200)):
            lines.append(f"L{i}.")
        return lines

    scroll = _draw_model_overlay(20, 140, [("row", [])], 0, [], 1, {}, True, "sessions",
                                 preview, True, preview_scroll=3, preview_from_top=True)
    out = ansi_strip(capsys.readouterr().out)
    assert scroll == 3
    assert "L3." in out and "L4." in out
    assert "L0." not in out and "L2." not in out


class _Screen:
    def __init__(self, tty_fd):
        self._tty_fd = tty_fd
        size = shutil.get_terminal_size()
        self._rows = size.lines
        self._cols = size.columns


def test_slash_filters_the_sessions_by_message_content(capsys):
    _save("plain", _turns("nothing here"), age=1)
    _save("target", _turns("hello", "the answer mentions quokka once"), age=50)
    index = _index()
    queries = []

    def _filter(pattern):
        queries.append(pattern)
        return index.search(pattern)

    master, slave = pty.openpty()
    closed = threading.Event()

    def type_keys():
        # '/', the word, Enter (keep the filter), Enter (pick the top row).
        # starts once the overlay is up.
        if closed.wait(0.3): return
        for key in "/quokka\r\r":
            os.write(master, key.encode())
            if closed.wait(0.08): return

    typist = threading.Thread(target=type_keys, daemon=True)
    typist.start()
    try:
        picked = prompt_tree_overlay(_Screen(slave),
                                     index.nodes,
                                     label_fn=lambda node: node["name"],
                                     preview_fn=lambda node, w, n: [],
                                     filter_fn=_filter,
                                     preview_from_top=True)
    finally:
        closed.set()
        typist.join(timeout=2)
        os.close(master)
        os.close(slave)
    assert picked == "target"                # the newer 'plain' session was filtered out
    assert "quokka" in queries
