"""Tests for the hover-widget layer: the Chip primitive (a styled pill with
optional position and timeout), the pending chip builder, the Tab-toggled
status widget, the layout's top-right widget painting with +N truncation, and
the screen's widget registry flattening."""
import re
import threading
import time

from cai.screen.ansi import ansi_strip, cur_move
from cai.screen.chip import Chip
from cai.screen.layout import Layout
from cai.screen.screen import Screen
from cai.tui import _pending_chip_lines, _Pending, _status_lines, _StatusWidget


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

    def add_widget(self, name, lines):
        self.widgets[name] = lines

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


# --- the status widget ---


def test_status_lines_sections_and_padding():
    lines = _status_lines("gpt-x", ["git"], ["fs__read"], [], 0, 0)
    plain = []
    for line in lines:
        plain.append(ansi_strip(line))
    assert plain[0].strip() == "model"
    assert plain[1].strip() == "gpt-x"
    assert plain[2].strip() == "skills"
    assert plain[3].strip() == "git"
    assert plain[4].strip() == "tools"
    assert plain[5].strip() == "fs__read"
    assert "sub-agents" not in " ".join(plain)       # empty section is left out
    widths = set()
    for text in plain:
        widths.add(len(text))
    assert len(widths) == 1


def test_status_lines_pending_and_subagents_and_empty():
    plain = []
    for line in _status_lines("", [], [], ["scout"], 2, 1):
        plain.append(ansi_strip(line).strip())
    assert plain == ["sub-agents", "scout", "pending", "1 steering", "2 queued"]
    assert ansi_strip(_status_lines("", [], [], [], 0, 0)[0]).strip() == "(nothing active)"


class _StatusClient:
    def __init__(self):
        self.model = "m1"
        self.skills = ["python"]
        self.tools = ["echo"]

    def get_info(self):
        return {"model": self.model}

    def get_selected_skills(self):
        return list(self.skills)

    def get_selected_tools(self):
        return list(self.tools)


def test_status_widget_toggles_and_refreshes_only_while_visible():
    host = _WidgetHost()
    pending = _Pending(host)
    client = _StatusClient()
    widget = _StatusWidget(host, client, pending)
    assert widget.visible is False
    widget.refresh()
    assert "status" not in host.widgets             # hidden: refresh paints nothing
    widget.toggle()
    assert widget.visible is True
    text = " ".join(ansi_strip(l) for l in host.widgets["status"])
    assert "m1" in text and "python" in text and "echo" in text
    client.skills = ["python", "fs"]
    widget.set_subagents(["scout"])
    pending.user_queued()
    widget.refresh()
    text = " ".join(ansi_strip(l) for l in host.widgets["status"])
    assert "fs" in text and "scout" in text and "1 queued" in text
    widget.toggle()
    assert widget.visible is False
    assert "status" not in host.widgets
