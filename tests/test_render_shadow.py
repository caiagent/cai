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


def _input_frame(layout, text, cursor_pos, cols=20):
    """render the prompt and return (cursor row, cursor col) from the frame."""
    import re
    buf = list(text)
    layout.update_input_height(buf, "> ", "  ")
    layout.render_input(buf, cursor_pos, Mode.INSERT, "> ", "  ", cols)
    moves = re.findall(r"\x1b\[(\d+);(\d+)H", layout_out())
    return int(moves[-1][0]), int(moves[-1][1])


def test_tall_prompt_scrolls_to_keep_the_cursor_above_the_status_line(capsys):
    global layout_out
    layout_out = lambda: capsys.readouterr().out
    layout = Layout(12, 20)                      # input area caps at 4 rows
    text = "\n".join(f"line{i}" for i in range(10))
    row, _ = _input_frame(layout, text, len(text))
    assert layout.input_height == 4
    assert row == layout.status_row - 1          # last input row, not the status row
    assert layout.input_scroll == 6
    # moving the cursor back to the top scrolls the window back with it
    row, col = _input_frame(layout, text, 0)
    assert (row, col) == (layout.input_start_row, 3)
    assert layout.input_scroll == 0
    # and deleting the text un-scrolls
    _input_frame(layout, "hi", 2)
    assert layout.input_scroll == 0


def test_cursor_after_a_full_row_gets_its_own_row(capsys):
    global layout_out
    layout_out = lambda: capsys.readouterr().out
    layout = Layout(12, 20)
    text = "x" * 18                              # "> " + 18 fills the row exactly
    row, col = _input_frame(layout, text, len(text))
    assert layout.input_height == 2
    assert (row, col) == (layout.status_row - 1, 1)
