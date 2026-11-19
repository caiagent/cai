"""Tests for message provenance (cai.origin): a harness-injected user turn
carries a local `_origin` key - set when a steer with an origin is folded in,
replaced by an envelope at the API boundary, and read by the views to tell a
sub-agent's report from a prompt the user typed."""
import io

from cai import tail
from cai.api import _wire_messages
from cai.environment import Environment, Settings
from cai.events import Event, EventType
from cai.llm import call_llm, SteerQueue
from cai.origin import envelope, origin_label, origin_of, split_origin, subagent_origin
from cai.screen.screen import Screen
from cai.tui import _Transcript
from cai.wire import Wire


# --- the origin strings -----------------------------------------------------

def test_subagent_origin_round_trips():
    origin = subagent_origin("audit")
    assert origin == "subagent:audit"
    assert split_origin(origin) == ("subagent", "audit")
    assert origin_label(origin) == "sub-agent 'audit'"


def test_origin_of_reads_only_tagged_messages():
    assert origin_of({"role": "user", "content": "hi"}) is None
    assert origin_of({"role": "user", "content": "r", "_origin": "subagent:a"}) == "subagent:a"
    assert origin_of("not a message") is None


def test_unknown_origin_kind_still_labels_and_envelopes():
    assert origin_label("cron:nightly") == "cron:nightly"
    text = envelope("cron:nightly", "body")
    assert "cron:nightly" in text and "Not written by the user" in text
    assert text.endswith("\n\nbody")


# --- the API boundary ---------------------------------------------------------

def test_wire_messages_envelope_a_tagged_message_and_strip_the_key():
    stored = {"role": "user", "content": "the findings", "_origin": "subagent:audit"}
    out = _wire_messages([stored])
    assert list(out[0].keys()) == ["role", "content"]
    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith("[sub-agent report]")
    assert "'audit'" in out[0]["content"]
    assert "not a message from the user" in out[0]["content"]
    assert out[0]["content"].endswith("\n\nthe findings")
    # the stored message keeps the raw report and its provenance
    assert stored == {"role": "user", "content": "the findings", "_origin": "subagent:audit"}


def test_wire_messages_leave_the_users_own_turn_alone():
    out = _wire_messages([{"role": "user", "content": "hi", "_display": "x"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_envelope_passes_non_text_content_through():
    parts = [{"type": "text", "text": "x"}]
    assert envelope("subagent:a", parts) is parts


# --- the queue and the loop ---------------------------------------------------

def test_steer_queue_keeps_origins_beside_plain_texts():
    q = SteerQueue()
    q.push("mine")
    q.push("report", "subagent:a")
    assert q.snapshot() == ["mine", "report"]
    assert q.remove(0, "mine")
    assert q.drain_entries() == [("report", "subagent:a")]
    assert q.drain_entries() == []


class _TextApi:
    def chat(self, messages, model, **kwargs):
        def gen():
            yield ("ok", None, None, {})
        return gen()


def test_call_llm_stores_the_origin_and_carries_it_on_the_event():
    pending = ["plain steer", ("the report", "subagent:a")]
    def steer():
        out = list(pending)
        pending.clear()
        return out
    messages = []
    events = list(call_llm(messages, "m", _TextApi(), steer=steer))
    assert messages[0] == {"role": "user", "content": "plain steer"}
    assert messages[1] == {"role": "user", "content": "the report", "_origin": "subagent:a"}
    users = []
    for event in events:
        if event.type == EventType.USER:
            users.append((event.text, event.origin))
    assert users == [("plain steer", None), ("the report", "subagent:a")]


def test_event_origin_survives_the_wire():
    event = Event(type=EventType.USER, text="r", origin="subagent:a")
    assert Wire.event_from_dict(Wire.event_to_dict(event)).origin == "subagent:a"
    plain = Event(type=EventType.USER, text="hi")
    assert Wire.event_from_dict(Wire.event_to_dict(plain)).origin is None


# --- the views ----------------------------------------------------------------

class _NoteScreen:
    def __init__(self):
        self.ops = []

    def write(self, text, kind=None, block=False):
        self.ops.append((text, kind))

    def content_width(self, kind=None):
        return 60


def test_transcript_paints_a_report_as_meta_not_as_a_prompt():
    screen = _NoteScreen()
    t = _Transcript(screen, Settings(), env=Environment())
    t.event(Event(type=EventType.USER, text="hi"))
    t.event(Event(type=EventType.USER, text="found it", origin="subagent:audit"))
    assert screen.ops[0] == ("> hi\n", Screen.USER)
    assert screen.ops[1] == ("[report from sub-agent 'audit']\nfound it\n", Screen.META)


def test_transcript_replay_reads_the_stored_origin():
    screen = _NoteScreen()
    t = _Transcript(screen, Settings(), env=Environment())
    t.replay([{"role": "user", "content": "found it", "_origin": "subagent:audit"}])
    assert screen.ops == [("[report from sub-agent 'audit']\nfound it\n", Screen.META)]


def test_tail_prints_a_report_under_its_header():
    out = io.StringIO()
    printer = tail._Printer(out)
    printer.event(Event(type=EventType.USER, text="hi"))
    printer.event(Event(type=EventType.USER, text="found it", origin="subagent:audit"))
    text = out.getvalue()
    assert "> hi" in text
    assert "[report from sub-agent 'audit']" in text
    assert "> found it" not in text
