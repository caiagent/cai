"""':merge N': another tab's work summarised into the tab in front. the
command is a bundled extension (builtins/extensions/merge), driven here the
way the TUI drives it - through a CommandContext over the real _Tabs, with
fake tabs - plus the ctx.tabs facade it is written against."""
import collections
import sys
import threading

import pytest

import cai
from cai import origin
from cai.commands import CommandContext, Tabs
from cai.environment import Environment
from cai.screen import Screen
from cai.tui import _Tabs


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def ext():
    """the bundled merge extension's module, loaded as any env loads it."""
    env = Environment().load()
    return sys.modules[env.commands()["merge"].fn.__module__]


@pytest.fixture
def summaries(ext, monkeypatch):
    """stands in for the model: records each request and streams `.events`,
    (event type, text) pairs - an exception among them is raised there."""
    class _Summaries:
        events = [("content", "did the "), ("content", "thing")]
        requests = []

        def __call__(self, request, model):
            self.requests.append((request, model))
            for event in self.events:
                if isinstance(event, BaseException):
                    raise event
                yield event

    out = _Summaries()
    out.requests = []
    monkeypatch.setattr(ext, "_summarise", out)
    return out


def _user(text):
    return {"role": "user", "content": text}


def _answer(text):
    return {"role": "assistant", "content": text}


class _Agent:
    def __init__(self, name, messages):
        self.name = name
        self.model = "m"
        self.messages = messages
        self.steers = []
        self.closed = False
        self._scratch = None
        self._scratch_owned = False

    def close(self):
        self.closed = True


class _Server:
    def __init__(self, agent):
        self.agent = agent


class _Client:
    def __init__(self, agent):
        self._agent = agent

    def steer(self, text, origin=None):
        self._agent.steers.append((text, origin))

    def get_messages(self):
        return list(self._agent.messages)

    def get_info(self):
        return {"name": self._agent.name, "model": self._agent.model}


class _FakeScreen:
    def __init__(self):
        self.written = []
        self.shown = []          # every state a rewritten block went through
        self.edits = []
        self.edited = None       # what the editor "returns": None = left unchanged
        self.aborted = 0
        self.busy = False

    def write(self, text, kind=None, block=False):
        self.written.append((text, kind))

    def write_block(self, text, kind=None):
        self.written.append((text, kind))
        self.shown.append(text)
        return len(self.written) - 1

    def replace_block(self, index, text, kind=None):
        self.written[index] = (text, kind)
        self.shown.append(text)

    def edit_in_editor(self, text, *, suffix='.txt'):
        self.edits.append(text)
        if callable(self.edited):
            return self.edited(text)
        return self.edited

    def abort_prompt(self):
        self.aborted += 1

    def foreground(self):
        self.foregrounded = True

    def set_tabline(self, cells, fill=''):
        pass

    def text(self):
        return "".join(text for text, _kind in self.written)


class _FakeTab:
    def __init__(self, agent):
        self.server = _Server(agent)
        self.client = _Client(agent)
        self.screen = _FakeScreen()
        self.stop = threading.Event()
        self.todo = collections.deque()
        self.closed = False
        self.unseen = False
        self.was_busy = False

    def close(self, restore_terminal=True):
        self.closed = True

    def marker(self):
        return ""

    def call_soon(self, fn):
        self.todo.append(fn)
        self.screen.abort_prompt()


def _tabs(*tabs, current=0):
    out = _Tabs()
    out._items = list(tabs)
    out._index = current
    return out


def _merge(ext, tabs, arg="2"):
    """type ':merge <arg>' in the tab in front."""
    facade = Tabs(tabs)
    ext.merge(CommandContext(arg, facade.current, facade))


def test_merge_is_a_bundled_extension_every_env_gets():
    first = Environment().load().commands()["merge"]
    second = Environment().load().commands()["merge"]
    assert first.takes_arg and not first.idle
    assert "builtins" in first.origin and second.fn is not first.fn
    assert Environment().commands() == {}                  # only load() brings it


def test_an_installed_merge_overrides_the_bundled_one(tmp_path):
    from cai.environment import extensions_dir
    bundle = tmp_path / "mine"
    (bundle).mkdir()
    (bundle / "init.py").write_text(
        "import cai\n\n@cai.command(help='mine')\ndef merge(ctx):\n    pass\n")
    import shutil
    shutil.copytree(bundle, f"{extensions_dir()}/mine")
    assert Environment().load().commands()["merge"].help == "mine"


def test_merge_point_is_the_common_prefix(ext):
    a = [_user("one"), _answer("two"), _user("three")]
    assert ext._merge_point(a, list(a)) == 3
    assert ext._merge_point(a, a[:2] + [_user("other")]) == 2
    assert ext._merge_point(a, []) == 0
    assert ext._merge_point(a, [_user("unrelated")]) == 0


def test_transcript_is_plain_text_with_tool_results_cut(ext):
    messages = [{"role": "system", "content": "sys"},
                _user("fix it"),
                {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "grep", "arguments": '{"q": "x"}'}}]},
                {"role": "tool", "content": "y" * 50},
                {"role": "user", "content": "report", "_origin": "subagent:scout"},
                _answer("done")]
    text = ext._merge_transcript(messages, 10)
    assert "sys" not in text
    assert "USER:\nfix it" in text
    assert 'ASSISTANT CALLED TOOL grep: {"q": "x"}' in text
    assert "TOOL RESULT:\n" + "y" * 10 + "… [40 more chars]" in text
    assert "REPORT FROM sub-agent 'scout':\nreport" in text
    assert "ASSISTANT:\ndone" in text


def test_request_separates_what_is_shared_from_the_work(ext):
    request = ext._merge_request([_user("shared prompt")], [_answer("new work")])
    shared, work = request.split("<later_work>")
    assert "shared prompt" in shared and "new work" not in shared
    assert "new work" in work


def test_an_unanswered_tool_call_is_not_work_to_summarise(ext):
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}, {"id": "2"}]}
    done = [_user("q"), _answer("a")]
    assert ext._settled(done + [call, {"role": "tool", "content": "r"}]) == done
    assert ext._settled(done + [call]) == done
    answered = done + [call, {"role": "tool", "content": "r"}, {"role": "tool", "content": "s"}]
    assert ext._settled(answered) == answered


def test_the_summary_call_is_a_bare_agent_streaming_the_merge_prompt(ext, monkeypatch):
    seen = {}

    class _Api:
        def chat(self, messages, model, **kwargs):
            seen["messages"] = messages
            seen["model"] = model
            seen["tools"] = kwargs.get("tools")
            assert kwargs.get("stream") is True
            yield None, "hmm", None, None
            yield "the ", None, None, None
            yield "summary", None, None, {}

    real = cai.Run
    monkeypatch.setattr(cai, "Run", lambda **kwargs: real(api=_Api(), **kwargs), raising=False)
    assert list(ext._summarise("the request", "m1")) == [("reasoning", "hmm"),
                                                         ("content", "the "),
                                                         ("content", "summary")]
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert seen["messages"][0]["content"] == ext._MERGE_SYSTEM
    assert seen["messages"][1]["content"] == "the request"
    assert seen["model"] == "m1" and not seen["tools"]


def test_merge_delivers_the_summary_tagged_with_the_source_session(ext, summaries):
    shared = [_user("start"), _answer("ok")]
    target = _FakeTab(_Agent("main", list(shared)))
    source = _FakeTab(_Agent("branch", shared + [_user("do x"), _answer("did x")]))
    tabs = _tabs(target, source)
    _merge(ext, tabs)
    request, model = summaries.requests[0]
    assert "did x" in request.split("<later_work>")[1] and model == "m"
    # the review happens in the tab in front - the one merged into - and at
    # once: nothing is handed to the main loop for later.
    assert not target.todo and target.screen.aborted == 0
    assert "summary of tab 2 to merge into this tab" in target.screen.edits[0]
    assert "did the thing" in target.screen.edits[0]
    assert target.server.agent.steers == [("did the thing", "tab:branch")]
    # one way: the merged tab is closed, the tab in front stays there.
    assert source.closed and source.server.agent.closed
    assert not target.closed
    assert tabs.count() == 1 and tabs.current is target


def test_the_summary_is_shown_as_it_is_written_then_folded_to_one_line(ext, summaries):
    shared = [_user("start")]
    target = _FakeTab(_Agent("main", list(shared)))
    source = _FakeTab(_Agent("branch", shared + [_answer("work")]))
    summaries.events = [("reasoning", "let me "), ("reasoning", "see"),
                        ("content", "did "), ("reasoning", "late thought"), ("content", "it")]
    tabs = _tabs(target, source)
    _merge(ext, tabs)
    heading = "[merging tab 2 — writing the summary, Ctrl-C cancels]"
    assert target.screen.shown == [heading,
                                   heading + "\nlet me ",
                                   heading + "\nlet me see",
                                   heading + "\ndid ",          # the answer replaces the thinking
                                   heading + "\ndid it",
                                   "[merged tab 2]"]
    # one block, rewritten in place: only its last state is left in the view.
    assert target.screen.written == [("[merged tab 2]", Screen.META)]
    assert target.server.agent.steers == [("did it", "tab:branch")]


def test_ctrl_c_cancels_the_merge_while_the_summary_is_written(ext, summaries):
    shared = [_user("start")]
    target = _FakeTab(_Agent("main", list(shared)))
    source = _FakeTab(_Agent("branch", shared + [_answer("work")]))
    summaries.events = [("content", "half a summ"), KeyboardInterrupt()]
    tabs = _tabs(target, source)
    _merge(ext, tabs)                                      # the interrupt stops here
    assert "half a summ" in target.screen.shown[-2]
    assert target.screen.written == [("[merge of tab 2 cancelled]", Screen.META)]
    assert target.screen.edits == [] and target.server.agent.steers == []
    assert not source.closed and tabs.count() == 2         # still open, to try again
    # and it can be merged again straight away.
    summaries.events = [("content", "all of it")]
    _merge(ext, tabs)
    assert target.server.agent.steers == [("all of it", "tab:branch")] and source.closed


def test_ctrl_c_closes_the_summary_call(ext, monkeypatch):
    closed = []
    interrupt = threading.Event()

    class _Run:
        def __init__(self, **kwargs):
            self.interrupt = interrupt

        def __iter__(self):
            yield cai.Event(type=cai.EventType.CONTENT, text="some")
            raise KeyboardInterrupt

        def close(self):
            closed.append(True)

    monkeypatch.setattr(cai, "Run", _Run, raising=False)
    with pytest.raises(KeyboardInterrupt):
        list(ext._summarise("request", "m"))
    assert closed == [True] and interrupt.is_set()


def test_merge_sends_the_reviewed_text_without_the_note(ext, summaries):
    shared = [_user("start")]
    target = _FakeTab(_Agent("main", list(shared)))
    source = _FakeTab(_Agent("branch", shared + [_answer("work")]))
    target.screen.edited = lambda text: text.replace("did the thing", "edited summary")
    tabs = _tabs(target, source)
    _merge(ext, tabs)
    assert target.server.agent.steers == [("edited summary", "tab:branch")]
    assert source.closed


def test_merge_into_a_running_tab_is_allowed(ext, summaries):
    shared = [_user("start")]
    target = _FakeTab(_Agent("main", list(shared)))
    source = _FakeTab(_Agent("branch", shared + [_answer("work")]))
    target.screen.busy = True                              # the steer folds into its run
    tabs = _tabs(target, source)
    _merge(ext, tabs)
    assert target.server.agent.steers == [("did the thing", "tab:branch")]
    assert source.closed


def test_emptying_the_review_cancels_the_merge(ext, summaries):
    target = _FakeTab(_Agent("main", []))
    source = _FakeTab(_Agent("branch", [_user("q"), _answer("a")]))
    target.screen.edited = ""
    tabs = _tabs(target, source)
    _merge(ext, tabs)
    assert target.server.agent.steers == []
    assert target.screen.written[-1][0] == "[merge of tab 2 cancelled]"
    assert not source.closed and tabs.count() == 2         # still open, to try again
    assert tabs.current is target


def test_merging_keeps_the_current_tab_in_front_whichever_side_closes(ext, summaries):
    shared = [_user("start")]
    # the merged tab sits before the current one: the index must follow it.
    source = _FakeTab(_Agent("branch", shared + [_answer("work")]))
    target = _FakeTab(_Agent("main", list(shared)))
    other = _FakeTab(_Agent("other", list(shared)))
    tabs = _tabs(source, target, other, current=1)
    _merge(ext, tabs, arg="1")
    assert source.closed and not other.closed
    assert tabs.count() == 2 and tabs.current is target
    # and after it.
    late = _FakeTab(_Agent("late", shared + [_answer("more")]))
    tabs = _tabs(target, other, late)
    _merge(ext, tabs, arg="3")
    assert late.closed
    assert tabs.count() == 2 and tabs.current is target


def test_merge_refuses_a_bad_source_and_reports_a_failed_summary(ext, summaries):
    target = _FakeTab(_Agent("main", []))
    source = _FakeTab(_Agent("branch", [_user("q"), _answer("a")]))
    summaries.events = [("content", "half"), RuntimeError("boom")]
    tabs = _tabs(target, source)
    _merge(ext, tabs, "9")
    _merge(ext, tabs, "x")
    _merge(ext, tabs, "0")
    _merge(ext, tabs, "1")
    text = target.screen.text()
    assert text.count("needs the number of an open tab") == 3
    assert "that is this tab" in text
    _merge(ext, tabs)
    assert target.screen.written[-1] == ("[error: merge summary failed: RuntimeError: boom]",
                                         Screen.ERROR)
    assert target.screen.edits == [] and target.server.agent.steers == []
    assert not source.closed
    summaries.events = [("reasoning", "thinking only")]
    _merge(ext, tabs)
    assert "the model returned no summary" in target.screen.text()
    assert target.server.agent.steers == [] and not source.closed


def test_merge_with_nothing_new_asks_for_no_summary(ext, summaries):
    shared = [_user("start"), _answer("ok")]
    target = _FakeTab(_Agent("main", list(shared)))
    source = _FakeTab(_Agent("branch", shared + [_user("unanswered")]))
    tabs = _tabs(target, source)
    _merge(ext, tabs)
    assert "nothing new to merge from tab 2" in target.screen.text()
    assert summaries.requests == [] and not source.closed


def test_merge_is_refused_while_the_merged_tab_is_running(ext, summaries):
    target = _FakeTab(_Agent("main", []))
    source = _FakeTab(_Agent("branch", [_user("q"), _answer("a")]))
    tabs = _tabs(target, source)
    source.screen.busy = True
    _merge(ext, tabs)
    assert "tab 2 is still running" in target.screen.text()
    assert summaries.requests == []                        # no summary was asked for
    assert target.server.agent.steers == [] and not source.closed


def test_command_context_client_and_screen_are_the_tabs():
    one = _FakeTab(_Agent("one", []))
    two = _FakeTab(_Agent("two", []))
    facade = Tabs(_tabs(one, two, current=1))
    ctx = CommandContext("arg", facade.current, facade)
    assert ctx.tab == ctx.tabs.current and ctx.tab.number == 2
    assert ctx.client is two.client and ctx.screen is two.screen
    ctx.write("hello")
    assert two.screen.text() == "hello"


# --- ctx.tabs: the facade a command gets ---

def test_tabs_facade_numbers_tabs_as_the_tab_line_does():
    one = _FakeTab(_Agent("one", []))
    two = _FakeTab(_Agent("two", []))
    tabs = _tabs(one, two, current=1)
    facade = Tabs(tabs)
    assert len(facade) == 2
    assert [tab.number for tab in facade] == [1, 2]
    assert facade.current == facade.at(2) and facade.current != facade.at(1)
    assert facade.at(0) is None and facade.at(3) is None
    assert facade.at(1).client is one.client and facade.at(1).screen is one.screen
    assert not hasattr(facade.at(1), "server")             # the wire only
    two.screen.busy = True
    assert facade.current.busy and not facade.at(1).busy


def test_tabs_facade_closes_a_tab_but_never_the_last():
    one = _FakeTab(_Agent("one", []))
    two = _FakeTab(_Agent("two", []))
    tabs = _tabs(one, two)
    facade = Tabs(tabs)
    second = facade.at(2)
    facade.close(second)
    assert two.closed and two.server.agent.closed and second.number == 0
    assert tabs.current is one and len(facade) == 1
    facade.close(second)                                   # already closed: nothing to do
    with pytest.raises(ValueError):
        facade.close(facade.current)
    assert not one.closed


def test_tab_call_soon_queues_for_the_main_thread_and_wakes_the_prompt():
    one = _FakeTab(_Agent("one", []))
    ran = []
    Tabs(_tabs(one)).current.call_soon(lambda: ran.append(1))
    assert one.screen.aborted == 1 and ran == []
    one.todo.popleft()()
    assert ran == [1]


def test_tab_origin_is_labelled_and_enveloped_as_merged_context():
    tag = origin.tab_origin("branch")
    assert origin.origin_label(tag) == "tab 'branch'"
    text = origin.envelope(tag, "the summary")
    assert text.endswith("\n\nthe summary")
    assert "merged session summary" in text and "'branch'" in text
