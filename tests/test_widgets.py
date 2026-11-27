"""Tests for the hover-widget layer: the Chip primitive (a styled pill with
optional position and timeout), the pending chip builder, the layout's top-right widget painting with +N
truncation, and the screen's widget registry flattening."""
import re
import threading
import time

from cai.screen.ansi import ansi_strip, cur_move
from cai.screen.chip import Chip
from cai.screen.layout import Layout
from cai.screen.screen import Screen
from cai.tui import _pending_chip_lines, _Pending, _agent_chip_lines, _AgentChips
from cai.tui import _status_rows, _status_panel_lines, _StatusPanel


# --- the Chip primitive ---


def test_chip_pads_its_text():
    lines = Chip("git").lines()
    assert [ansi_strip(l) for l in lines] == [" git "]


def test_chip_styles_the_body():
    line = Chip("git", sgr="\033[36m").lines()[0]
    assert line.startswith("\033[36m")


def test_chip_defaults():
    chip = Chip("git")
    assert chip.position is None
    assert chip.timeout is None


# --- the pending chip builder ---


def test_pending_chip_lines_empty_when_both_zero():
    assert _pending_chip_lines(0, 0) == []


def test_pending_chip_lines_steer_above_user():
    lines = _pending_chip_lines(1, 2)
    assert len(lines) == 2
    assert "2 steering" in ansi_strip(lines[0])
    assert "1 queued" in ansi_strip(lines[1])


def test_pending_chip_lines_one_row_each():
    assert len(_pending_chip_lines(3, 0)) == 1
    assert "3 queued" in ansi_strip(_pending_chip_lines(3, 0)[0])
    assert len(_pending_chip_lines(0, 4)) == 1
    assert "4 steering" in ansi_strip(_pending_chip_lines(0, 4)[0])


def test_pending_chip_lines_rows_share_a_width():
    lines = _pending_chip_lines(1, 10)
    assert len(ansi_strip(lines[0])) == len(ansi_strip(lines[1]))


# --- the _Pending widget owner ---


class _WidgetHost:
    """a Screen stand-in that just records the last add/remove of a widget."""

    def __init__(self):
        self.widgets = {}

    def add_widget(self, name, lines, bottom=False, center=False, dismiss=None):
        self.widgets[name] = lines
        self.dismiss = dismiss

    def viewport_size(self):
        return 40, 100

    def remove_widget(self, name):
        self.widgets.pop(name, None)


def test_pending_counts_user_turns_up_and_down():
    host = _WidgetHost()
    pending = _Pending(host)
    pending.user_queued()
    pending.user_queued()
    assert "2 queued" in ansi_strip(host.widgets["pending"][0])
    pending.user_started()
    assert "1 queued" in ansi_strip(host.widgets["pending"][0])
    pending.user_started()
    assert "pending" not in host.widgets


def test_pending_user_started_never_goes_negative():
    host = _WidgetHost()
    pending = _Pending(host)
    pending.user_started()
    assert "pending" not in host.widgets


def test_pending_tracks_the_steer_count():
    host = _WidgetHost()
    pending = _Pending(host)
    pending.set_steer(3)
    assert "3 steering" in ansi_strip(host.widgets["pending"][0])
    pending.set_steer(0)
    assert "pending" not in host.widgets


# --- chips on the screen: stacking, anchoring, timeout ---


class _ChipHost:
    """a Screen stand-in with just the chip registry state. the focus stack
    is parked off 'main' so add/remove skip the terminal repaint."""

    add_chip = Screen.add_chip
    remove_chip = Screen.remove_chip
    _positioned_cells = Screen._positioned_cells

    def __init__(self):
        self._widgets = {}
        self._positioned_chips = {}
        self._bottom_widgets = {}
        self._center_widgets = {}
        self._key_dismiss = {}
        self._chip_timers = {}
        self._render_lock = threading.RLock()
        self._focus_stack = ['overlay']


def test_add_chip_without_position_joins_the_widget_stack():
    host = _ChipHost()
    host.add_chip("note", Chip("hi"))
    assert host._widgets["note"] == Chip("hi").lines()
    assert host._positioned_chips == {}


def test_add_chip_with_position_anchors_its_cells():
    host = _ChipHost()
    host.add_chip("note", Chip("hi", position=(5, 2)))
    assert "note" not in host._widgets
    cells = host._positioned_cells()
    assert [(vrow, col) for vrow, col, _ in cells] == [(5, 2)]


def test_add_chip_replacement_can_move_between_stack_and_anchor():
    host = _ChipHost()
    host.add_chip("note", Chip("hi"))
    host.add_chip("note", Chip("hi", position=(1, 1)))
    assert "note" not in host._widgets
    assert "note" in host._positioned_chips


def test_remove_chip_forgets_both_registries():
    host = _ChipHost()
    host.add_chip("stacked", Chip("a"))
    host.add_chip("anchored", Chip("b", position=(0, 0)))
    host.remove_chip("stacked")
    host.remove_chip("anchored")
    host.remove_chip("unknown")
    assert host._widgets == {}
    assert host._positioned_chips == {}


def test_chip_timeout_removes_it():
    host = _ChipHost()
    host.add_chip("note", Chip("hi", timeout=0.05))
    assert "note" in host._widgets
    deadline = time.monotonic() + 2
    while "note" in host._widgets and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "note" not in host._widgets
    assert host._chip_timers == {}


def test_replacing_a_chip_cancels_the_old_timeout():
    host = _ChipHost()
    host.add_chip("note", Chip("hi", timeout=0.05))
    host.add_chip("note", Chip("hi"))
    time.sleep(0.2)
    assert "note" in host._widgets


def test_render_content_paints_positioned_cells(capsys):
    layout = Layout(10, 40)
    layout.render_content(["hello"], 8, 40,
                          positioned_cells=[(2, 5, "chip")])
    out = capsys.readouterr().out
    # 0-based (row 2, col 5) lands at 1-based terminal (3, 6)
    assert cur_move(3, 6) in out
    assert "chip" in out


def test_render_content_clips_positioned_cells(capsys):
    layout = Layout(10, 40)
    layout.render_content(["hello"], 8, 40,
                          positioned_cells=[(20, 0, "below"),
                                            (0, 38, "overflow")])
    out = capsys.readouterr().out
    assert "below" not in out
    assert "ov" in out
    assert "overflow" not in out


# --- :prompts entries ---

from cai.tui import _prompt_entries


def test_prompt_entries_dedupe_keeps_newest_first():
    history = ["fix the bug", "add tests", "fix the bug"]
    entries = _prompt_entries(history)
    assert list(entries) == ["fix the bug", "add tests"]


def test_prompt_entries_flatten_multiline_labels():
    history = ["line one\n  line two"]
    entries = _prompt_entries(history)
    assert list(entries) == ["line one line two"]
    assert entries["line one line two"] == "line one\n  line two"


def test_prompt_entries_skip_blanks():
    assert _prompt_entries(["", "  \n ", "real"]) == {"real": "real"}



# --- the sub-agent chips ---

def test_agent_chip_lines_one_padded_pill_per_agent():
    assert _agent_chip_lines([]) == []
    lines = _agent_chip_lines(["scout", "audit-auth-flow"])
    plain = []
    for line in lines:
        plain.append(ansi_strip(line))
    assert plain == [" scout           ", " audit-auth-flow "]


def test_agent_chips_follow_the_running_names():
    host = _WidgetHost()
    chips = _AgentChips(host)
    chips.set_names([])
    assert "agents" not in host.widgets            # idle: nothing painted
    chips.set_names(["scout"])
    assert ansi_strip(host.widgets["agents"][0]).strip() == "scout"
    painted = host.widgets["agents"]
    chips.set_names(["scout"])
    assert host.widgets["agents"] is painted       # unchanged: no repaint
    chips.set_names(["scout", "audit"])
    assert len(host.widgets["agents"]) == 2
    chips.set_names([])
    assert "agents" not in host.widgets            # all finished: chip gone


def test_agent_chips_are_a_bottom_widget_like_pending():
    calls = []

    class _Host:
        def add_widget(self, name, lines, bottom=False):
            calls.append((name, bottom))

        def remove_widget(self, name):
            pass
    _AgentChips(_Host()).set_names(["scout"])
    assert calls == [("agents", True)]


# --- the status panel: a framed window in the middle of the viewport ---


def _panel(max_rows=40, max_cols=100, width=60, **overrides):
    fields = {
        "model": "gpt-x",
        "cwd": "/work/project",
        "skills": ["fs-read-only", "subagents"],
        "skill_tools": {"fs-read-only": ["grep", "read_file"], "subagents": ["launch_agent"]},
        "tools": ["grep", "launch_agent", "python", "read_file"],
        "subagents": [],
        "allowed": [],
        "disallowed": [],
        "user": 0,
        "steer": 0,
    }
    fields.update(overrides)
    rows = _status_rows(fields["model"],
                        fields["cwd"],
                        fields["skills"],
                        fields["skill_tools"],
                        fields["tools"],
                        fields["subagents"],
                        fields["allowed"],
                        fields["disallowed"],
                        fields["user"],
                        fields["steer"],
                        width)
    lines = _status_panel_lines(rows, max_rows, max_cols)
    plain = []
    for line in lines:
        plain.append(ansi_strip(line))
    return plain


def test_status_panel_is_a_closed_frame_of_one_width():
    plain = _panel()
    assert plain[0].startswith("╭─ status ")
    assert plain[0].endswith("╮")
    assert plain[-1].startswith("╰")
    assert plain[-1].endswith("╯")
    for line in plain[1:-1]:
        assert line.startswith("│")
        assert line.endswith("│")
    assert len({len(line) for line in plain}) == 1


def test_status_panel_lists_what_the_status_view_listed():
    body = "\n".join(_panel(subagents=["scout"],
                            allowed=["/data"],
                            disallowed=["/secret"],
                            user=1,
                            steer=2))
    for needle in ("gpt-x", "/work/project", "fs-read-only", "subagents",
                   "grep", "read_file", "scout", "/data", "/secret",
                   "2 steering", "1 queued"):
        assert needle in body


def test_status_panel_groups_each_skill_with_its_tools():
    plain = _panel()
    index = {}
    for i, line in enumerate(plain):
        for name in ("◆ fs-read-only", "◆ subagents", "grep", "launch_agent", "python"):
            if name in line:
                index.setdefault(name, i)
    assert index["◆ fs-read-only"] < index["grep"] < index["◆ subagents"] < index["launch_agent"]
    assert "read_file" in plain[index["grep"]]         # a skill's tools share its block
    # a tool no skill declares sits under its own 'tools' label, after the skills
    assert "tools 1" in plain[index["python"]]
    assert index["python"] > index["launch_agent"]


def test_status_panel_hides_a_skill_tool_that_was_deselected():
    body = "\n".join(_panel(tools=["grep"]))
    assert "grep" in body
    assert "read_file" not in body
    assert "launch_agent" not in body
    assert "◆ subagents" in body                       # the skill itself still shows


def test_status_panel_colours_skills_and_tools_differently():
    rows = _status_rows("m", "/w", ["s"], {"s": ["t"]}, ["t"], [], [], [], 0, 0, 60)
    styles = {}
    for row in rows:
        for text, sgr in row:
            styles[text] = sgr
    assert styles["s"] != styles["t"]
    assert styles["s"] and styles["t"]


def test_status_panel_leaves_out_empty_sections():
    body = "\n".join(_panel())
    for label in ("agents ", "allowed", "denied", "pending"):
        assert label not in body.replace("subagents", "")


def test_status_panel_wraps_long_name_lists_inside_the_frame():
    tools = []
    for i in range(40):
        tools.append(f"tool_number_{i}")
    plain = _panel(tools=tools, max_cols=70, width=60)
    assert len(plain[0]) <= 70
    assert len({len(line) for line in plain}) == 1
    assert "tool_number_39" in "\n".join(plain)


def test_status_panel_cuts_a_body_taller_than_the_viewport():
    tools = []
    for i in range(200):
        tools.append(f"tool_number_{i}")
    plain = _panel(tools=tools, max_rows=12)
    assert len(plain) == 12
    assert "more" in plain[-3]


def test_status_panel_keeps_the_end_of_a_long_cwd():
    body = "\n".join(_panel(cwd="/a" * 80 + "/leaf", width=40))
    assert "/leaf" in body
    assert "…" in body


def test_status_panel_gives_up_when_the_viewport_is_too_small():
    assert _panel(max_cols=20) == []
    assert _panel(max_rows=4) == []


class _PanelClient:
    def get_info(self):
        return {"model": "gpt-x", "cwd": "/work"}

    def get_selected_skills(self):
        return ["subagents"]

    def get_skill_tools(self):
        return {"subagents": ["grep"]}

    def get_selected_tools(self):
        return ["grep"]

    def get_paths(self):
        return {"allowed": [], "disallowed": []}


def test_status_panel_starts_hidden_and_toggles():
    host = _WidgetHost()
    panel = _StatusPanel(host, _PanelClient(), _Pending(host))
    panel.refresh()
    assert "status" not in host.widgets            # hidden: refresh paints nothing
    panel.toggle()
    assert "gpt-x" in ansi_strip("\n".join(host.widgets["status"]))
    panel.set_subagents(["scout"])
    panel.refresh()
    assert "scout" in ansi_strip("\n".join(host.widgets["status"]))
    panel.toggle()
    assert "status" not in host.widgets


def test_status_panel_hides_on_a_keypress_and_stays_hidden():
    host = _WidgetHost()
    panel = _StatusPanel(host, _PanelClient(), _Pending(host))
    panel.toggle()
    host.dismiss()                                 # what the screen calls on a key
    assert "status" not in host.widgets
    assert not panel.visible
    panel.refresh()                                # the poll loop's next tick
    assert "status" not in host.widgets
    panel.toggle()                                 # one Tab brings it back
    assert "status" in host.widgets


def test_screen_dismisses_a_widget_on_any_key_and_swallows_tab_and_esc():
    host = _ChipHost()
    host.add_widget = Screen.add_widget.__get__(host)
    host.remove_widget = Screen.remove_widget.__get__(host)
    host._dismiss_on_key = Screen._dismiss_on_key.__get__(host)
    for key, swallowed in (("\t", True), ("\033", True), ("j", False)):
        host.add_widget("status", ["abcd"], center=True,
                        dismiss=lambda: host.remove_widget("status"))
        assert host._dismiss_on_key(key) is swallowed
        assert "status" not in host._center_widgets
        assert host._key_dismiss == {}
    assert host._dismiss_on_key("j") is False      # nothing left to dismiss


def test_center_widget_is_placed_in_the_middle_and_painted_last():
    host = _ChipHost()
    host.add_widget = Screen.add_widget.__get__(host)
    host._layout = Layout(rows=24, cols=80)
    host.add_widget("pending", ["bottom"], bottom=True)
    host.add_widget("status", ["abcd", "efgh"], center=True)
    cells = host._positioned_cells()
    rows = host._layout.content_rows
    assert cells[-2:] == [((rows - 2) // 2, 38, "abcd"), ((rows - 2) // 2 + 1, 38, "efgh")]


def test_agent_errors_are_written_as_error_blocks():
    from cai.tui import _show_agent_errors

    written = []

    class _Screen:
        def write(self, text, kind=None, block=False):
            written.append((text, kind, block))

    class _Client:
        def drain_errors(self):
            return ["failed loading MCP tool 'a__b': boom"]

    _show_agent_errors(_Screen(), _Client())
    assert written == [("[error: failed loading MCP tool 'a__b': boom]\n", Screen.ERROR, True)]


def test_environment_errors_drain_once_and_do_not_repeat_while_queued():
    from cai.environment import Environment

    env = Environment()
    env.report_error("hook 'h' raised: boom")
    env.report_error("hook 'h' raised: boom")      # still undrained: not queued twice
    env.report_error("other")
    assert env.drain_errors() == ["hook 'h' raised: boom", "other"]
    assert env.drain_errors() == []
    env.report_error("other")                      # after a drain it shows again
    assert env.drain_errors() == ["other"]


# --- tabs ---


def test_clone_point_drops_a_tool_exchange_still_in_flight():
    from cai.tui import _clone_point

    user = {"role": "user", "content": "go"}
    calls = {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]}
    first = {"role": "tool", "tool_call_id": "a", "content": "1"}
    second = {"role": "tool", "tool_call_id": "b", "content": "2"}
    assert _clone_point([user, calls]) == [user]                     # none answered yet
    assert _clone_point([user, calls, first]) == [user]              # one of two answered
    assert _clone_point([user, calls, first, second]) == [user, calls, first, second]
    answer = {"role": "assistant", "content": "done"}
    assert _clone_point([user, answer]) == [user, answer]
    assert _clone_point([]) == []


def test_clone_point_is_a_copy():
    from cai.tui import _clone_point

    messages = [{"role": "user", "content": "go"}]
    copied = _clone_point(messages)
    copied[0]["content"] = "changed"
    assert messages[0]["content"] == "go"


def test_tab_line_numbers_labels_and_marks_the_tabs():
    from cai.tui import _tab_line_cells, _TAB_CURRENT, _TAB_OTHER

    cells = _tab_line_cells(["fix bug", "abc123", "plan"], ["", "…", "+"], 1)
    assert cells == [(" 1 fix bug ", _TAB_OTHER),
                     (" 2… abc123 ", _TAB_CURRENT),
                     (" 3+ plan ", _TAB_OTHER)]


def test_tab_label_is_the_first_typed_prompt():
    from cai.tui import _tab_label

    assert _tab_label([], "abc123") == "abc123"
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": "  fix the\n login bug please, it is broken  "}]
    assert _tab_label(messages, "abc123") == "fix the login bug pleas…"


def test_tab_line_takes_the_top_row_and_shifts_the_viewport():
    layout = Layout(rows=24, cols=20)
    rows = layout.content_rows
    assert layout.content_top == 0
    assert layout.set_tabline([(" 1 a ", "<A>"), (" 2 b ", "<B>")], "<F>") is True
    assert layout.content_top == 1
    assert layout.content_rows == rows - 1
    assert ansi_strip(layout._tabline_text(20)).replace("<A>", "").replace("<B>", "").replace("<F>", "") \
        == " 1 a  2 b " + " " * 10
    assert layout._tabline_text(7).count(" 2") == 1          # cut, never wider than the row
    assert layout.set_tabline(None) is True
    assert layout.content_rows == rows


def test_normal_mode_tab_keys_raise_the_tab_commands():
    from cai.screen.modes import ModeHandler
    from cai.screen.state import CommandException, Mode, TUIState

    class _Screen:
        _live = None
        _buffer = None
        _layout = None

    handler = ModeHandler()
    for keys, command in (("gt", "tabnext"), ("gT", "tabprev"), ("\x17T", "checkout")):
        state = TUIState()
        state.mode = Mode.NORMAL
        raised = None
        try:
            for key in keys:
                handler.handle_key(key, state, _Screen())
        except CommandException as e:
            raised = e.value
        assert raised == command
        assert state.pending_key == ''


def test_a_background_screen_takes_writes_without_painting():
    host = _ChipHost()
    host._focus_stack = ['main']
    host.push_focus = Screen.push_focus.__get__(host)
    host.background = Screen.background.__get__(host)
    host.background()
    assert host._focus_stack == ['main', 'background']
    host.add_widget = Screen.add_widget.__get__(host)
    host.add_widget("tabs", ["x"])                 # would repaint if in front
    assert host._widgets["tabs"] == ["x"]


# --- normal is the resting mode ---


def test_a_fresh_state_starts_in_normal_mode():
    from cai.screen.state import Mode, TUIState

    assert TUIState().mode == Mode.NORMAL


def test_submitting_a_prompt_returns_to_normal_mode():
    from cai.screen.modes import ModeHandler
    from cai.screen.state import Mode, SubmitException, TUIState

    class _Screen:
        _cursor_pos = 2
        _in_prompt = True

        def __init__(self, text):
            self._input_buf = list(text)
            self._history = []

        def _save_history_entry(self, entry):
            pass

    for text, mode in (("hi", Mode.NORMAL), ("  ", Mode.INSERT)):
        state = TUIState()
        state.mode = Mode.INSERT
        try:
            ModeHandler()._handle_submit(state, _Screen(text))
        except SubmitException as e:
            assert e.value == text
        assert state.mode == mode                  # an empty Enter sends nothing: stay put


# --- the `:`-command table ----------------------------------------------------

class _CmdScreen:
    def __init__(self, busy=False):
        self.busy = busy
        self.written = []

    def write(self, text, kind=None, block=False):
        self.written.append(text)


class _CmdTab:
    def __init__(self, busy=False):
        self.screen = _CmdScreen(busy)


class _CmdTabs:
    def __init__(self, tab):
        self.current = tab


def test_every_command_is_defined_once_and_feeds_the_palette():
    from cai.tui import _COMMANDS, _PALETTE_ARG_COMMANDS, _PALETTE_COMMANDS, _command_index
    names = [command.name for command in _COMMANDS]
    assert len(names) == len(set(names))
    assert [name for name, _help in _PALETTE_COMMANDS] == names
    assert _PALETTE_ARG_COMMANDS == ("allow", "disallow", "save", "load")
    index = _command_index(_COMMANDS)
    assert index["q"] is index["quit"] and index["qall"] is index["qa"]


def test_dispatch_runs_the_command_with_its_argument():
    from cai.tui import _Command, _command_index, _dispatch
    seen = []
    index = _command_index([_Command("go", "", lambda ctx: seen.append((ctx.head, ctx.arg))),
                            _Command("bye", "", lambda ctx: True, aliases=("b",))])
    tabs = _CmdTabs(_CmdTab())
    assert _dispatch(index, tabs, "go  far away ") is False
    assert seen == [("go", "far away")]
    assert _dispatch(index, tabs, "b") is True              # True leaves the TUI


def test_dispatch_refuses_an_idle_only_command_while_busy():
    from cai.tui import _Command, _command_index, _dispatch
    ran = []
    index = _command_index([_Command("clear", "", lambda ctx: ran.append(1), idle=True)])
    tab = _CmdTab(busy=True)
    assert _dispatch(index, _CmdTabs(tab), "clear") is False
    assert ran == [] and tab.screen.written == ["[busy — :clear when idle]\n"]
    tab.screen.busy = False
    _dispatch(index, _CmdTabs(tab), "clear")
    assert ran == [1]


def test_dispatch_falls_back_to_an_extension_command_then_reports_unknown():
    from cai.environment import Environment
    from cai.commands import Command
    from cai.tui import _dispatch

    class _Shared:
        env = Environment()

    class _Worker:
        ui = None

    tab = _CmdTab()
    tab.shared = _Shared()
    tab.worker = _Worker()
    tab.client = object()
    got = []
    _Shared.env.register_command("hello", Command(fn=lambda ctx: got.append(ctx.args)))
    assert _dispatch({}, _CmdTabs(tab), "hello there") is False
    assert got == ["there"]
    _dispatch({}, _CmdTabs(tab), "nope")
    assert tab.screen.written == ["[unknown command: :nope]\n"]


def test_an_extension_command_gets_the_tabs_and_honours_idle():
    import cai
    from cai.environment import Environment
    from cai.tui import _dispatch

    class _Shared:
        env = Environment.default()

    class _Worker:
        ui = None

    tab = _CmdTab()
    tab.shared = _Shared()
    tab.worker = _Worker()
    tab.client = object()
    got = []

    @cai.command(takes_arg=True, idle=True)
    def careful(ctx):
        got.append(ctx)

    registered = Environment.default().commands()["careful"]
    assert registered.takes_arg and registered.idle
    tabs = _CmdTabs(tab)
    tab.screen.busy = True
    assert _dispatch({}, tabs, "careful") is False
    assert got == [] and tab.screen.written == ["[busy — :careful when idle]\n"]
    tab.screen.busy = False
    _dispatch({}, tabs, "careful")
    assert len(got) == 1 and got[0].tabs._tabs is tabs     # ctx.tabs wraps the TUI's tabs
    assert got[0].tab == got[0].tabs.current
    assert got[0].screen is tab.screen and got[0].client is tab.client


def test_ctrl_c_in_an_extension_command_ends_the_command_not_the_tui():
    import cai
    from cai.environment import Environment
    from cai.tui import _dispatch

    class _Shared:
        env = Environment.default()

    class _Worker:
        ui = None

    tab = _CmdTab()
    tab.shared = _Shared()
    tab.worker = _Worker()
    tab.client = object()

    @cai.command
    def slow(ctx):
        raise KeyboardInterrupt

    assert _dispatch({}, _CmdTabs(tab), "slow") is False    # False: the TUI carries on
    assert tab.screen.written == ["[:slow interrupted]\n"]



# ── :prompts word search ─────────────────────────────────────────────────────

from cai.screen.overlays.model import _filter_and_sort, _terms_match


def test_terms_match_needs_every_word_as_a_substring():
    # a scattered-letter subsequence ("fx" in "fix") is not a hit
    assert _terms_match("fx", "fix the tui crash")[0] is False
    assert _terms_match("crash fix", "fix the tui crash")[0] is True
    assert _terms_match("crash sandbox", "fix the tui crash")[0] is False
    assert _terms_match("TUI", "fix the tui crash") == (True, 0, [8, 9, 10])


def test_prompt_search_keeps_newest_first_and_ranks_phrase_hits_first():
    entries = ["crash in the tui", "tui crash on resize", "add themes", "the tui crash again"]
    names = [name for name, _ in _filter_and_sort(entries, "tui crash", None, _terms_match)]
    assert names == ["tui crash on resize", "the tui crash again", "crash in the tui"]
