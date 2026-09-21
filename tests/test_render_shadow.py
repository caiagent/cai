"""the layout's row shadow: a render emits only the rows that changed since
the last paint, and paints text before erasing the rest of the row (no
blank-then-refill flash on terminals without synchronized output)."""
from cai.screen.ansi import ERASE_LINE, SGR_RESET, cur_move
from cai.screen.layout import Layout
from cai.screen.state import Mode


def test_unchanged_rows_are_not_repainted(capsys):
    layout = Layout(10, 40)
    layout.render_content(["a", "b"], 8, 40)
    first = capsys.readouterr().out
    assert cur_move(1, 1) in first
    assert cur_move(8, 1) in first
    layout.render_content(["a", "b"], 8, 40)
    assert capsys.readouterr().out == ""


def test_only_the_changed_row_is_repainted(capsys):
    layout = Layout(10, 40)
    layout.render_content(["a", "b"], 8, 40)
    capsys.readouterr()
    layout.render_content(["a", "bc"], 8, 40)
    out = capsys.readouterr().out
    assert cur_move(2, 1) in out
    assert cur_move(1, 1) not in out
    assert cur_move(3, 1) not in out


def test_text_lands_before_the_erase(capsys):
    layout = Layout(10, 40)
    layout.render_content(["hello"], 8, 40)
    out = capsys.readouterr().out
    row = out[out.index(cur_move(1, 1)):out.index(cur_move(2, 1))]
    assert row == cur_move(1, 1) + "hello" + SGR_RESET + ERASE_LINE


def test_full_width_row_skips_the_erase(capsys):
    layout = Layout(10, 5)
    layout.render_content(["abcde"], 8, 5)
    out = capsys.readouterr().out
    row = out[out.index(cur_move(1, 1)):out.index(cur_move(2, 1))]
    assert ERASE_LINE not in row


def test_invalidate_repaints_everything(capsys):
    layout = Layout(10, 40)
    layout.render_content(["a"], 8, 40)
    capsys.readouterr()
    layout.invalidate()
    layout.render_content(["a"], 8, 40)
    out = capsys.readouterr().out
    assert cur_move(1, 1) in out
    assert cur_move(8, 1) in out


def test_resize_invalidates(capsys):
    layout = Layout(10, 40)
    layout.render_content(["a"], 8, 40)
    capsys.readouterr()
    layout.resize(10, 40)
    layout.render_content(["a"], 8, 40)
    assert cur_move(1, 1) in capsys.readouterr().out


def test_widget_change_repaints_its_row(capsys):
    layout = Layout(10, 40)
    layout.render_content(["a"], 8, 40, widget_lines=["w1"])
    capsys.readouterr()
    layout.render_content(["a"], 8, 40, widget_lines=["w2"])
    out = capsys.readouterr().out
    assert "w2" in out
    assert cur_move(2, 1) not in out


def test_status_and_input_rows_go_through_the_shadow(capsys):
    layout = Layout(10, 40)
    layout.render_status(Mode.NORMAL, "m", cols=40)
    layout.render_input([], 0, Mode.NORMAL, "> ", "  ", 40)
    capsys.readouterr()
    layout.render_status(Mode.NORMAL, "m", cols=40)
    layout.render_input([], 0, Mode.NORMAL, "> ", "  ", 40)
    assert capsys.readouterr().out == ""
    layout.render_status(Mode.NORMAL, "changed", cols=40)
    assert cur_move(10, 1) in capsys.readouterr().out
