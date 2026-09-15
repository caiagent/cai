"""Interactive tools toggle overlay (alternate screen, floating box). the list
itself - cursor, checks, search - is a ToggleList; this file only frames it in
a centered box and runs the key loop."""

import select
import shutil
import signal
import sys
import termios
import tty

from ..ansi import (
    ALT_ENTER, ALT_EXIT,
    CUR_SHOW, CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE,
    cur_move,
)
from ..input import read_key, parse_mouse
from ..togglelist import ToggleList


def _emit_diff(new_lines, prev_lines, start_c, first_draw):
    """build the escape string to update only changed rows."""
    out = []
    if first_draw:
        sys.stdout.write(ERASE_SCREEN)
        for row_off, (r, text) in new_lines.items():
            out.append(f'{cur_move(r, start_c)}{text}')
    else:
        for row_off, (r, text) in new_lines.items():
            if prev_lines.get(row_off) != text:
                out.append(f'{cur_move(r, start_c)}{text}')
    prev_lines.clear()
    for row_off, (r, text) in new_lines.items():
        prev_lines[row_off] = text
    return ''.join(out)


def _box_geometry(rows, cols):
    """(inner_w, visible_n, start_r, start_c) of the centered box."""
    inner_w = max(20, int(cols * 0.95) - 2)
    box_w = inner_w + 2
    overhead = 4
    max_box_h = max(overhead + 1, int(rows * 0.95))
    visible_n = max_box_h - overhead
    box_h = visible_n + overhead
    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)
    return inner_w, visible_n, start_r, start_c


def draw_tools_overlay(rows, cols, lst, prev_lines, first_draw, title=''):
    """render the centered floating box around a ToggleList (pure draw, no
    event loop)."""
    inner_w, visible_n, start_r, start_c = _box_geometry(rows, cols)

    H = '─'
    TL = '┌'
    TR = '┐'
    BL = '└'
    BR = '┘'
    VL = '│'
    ML = '├'
    MR = '┤'
    h_line = H * inner_w

    new_lines = {}

    def put(row_off, text):
        r = start_r + row_off
        if 1 <= r <= rows:
            new_lines[row_off] = (r, text)

    top = h_line
    if title:
        top = f'{H} {title} '[:inner_w].ljust(inner_w, H)
    put(0, f'{TL}{top}{TR}')

    for i, row in enumerate(lst.rows(inner_w, visible_n)):
        put(1 + i, f'{VL}{row}{VL}')

    put(1 + visible_n, f'{ML}{h_line}{MR}')

    status_cell = lst.status(inner_w)[:inner_w].ljust(inner_w)
    if lst.search_mode:
        put(1 + visible_n + 1, f'{VL}{SGR_REVERSE}{status_cell}{SGR_RESET}{VL}')
    else:
        put(1 + visible_n + 1, f'{VL}{status_cell}{VL}')

    put(1 + visible_n + 2, f'{BL}{h_line}{BR}')

    out = _emit_diff(new_lines, prev_lines, start_c, first_draw)

    if lst.search_mode:
        search_text = ''.join(lst.search_buf)
        cursor_col = start_c + 1 + 1 + 1 + len(search_text)
        cursor_row = start_r + 1 + visible_n + 1
        out += f'{CUR_SHOW}{cur_move(cursor_row, cursor_col)}'
    else:
        out += CUR_HIDE

    sys.stdout.write(out)
    sys.stdout.flush()


def prompt_tools_overlay(screen, tool_entries, enabled, title=''):
    """interactive tools toggle overlay.

    tool_entries : list of (name, origin_label) tuples
    enabled      : set of currently enabled tool names

    navigation : j / k / arrows / Ctrl-U / Ctrl-D / gg / G
    toggle     : Space
    search fwd : /pattern  then n / N to cycle
    search bwd : ?pattern  then N / n to cycle
    close      : ESC or Enter"""
    if not tool_entries:
        return set(enabled)

    lst = ToggleList(tool_entries, enabled)
    prev_lines = {}
    first_draw = [True]
    resize_pending = [False]

    def _on_resize(signum, frame):
        ts = shutil.get_terminal_size()
        screen._rows, screen._cols = ts.lines, ts.columns
        resize_pending[0] = True
        first_draw[0] = True

    def _redraw():
        draw_tools_overlay(screen._rows, screen._cols, lst, prev_lines, first_draw[0], title=title)
        first_draw[0] = False

    old_attrs = termios.tcgetattr(screen._tty_fd)
    orig_handler = signal.getsignal(signal.SIGWINCH)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    try:
        signal.signal(signal.SIGWINCH, _on_resize)
        tty.setraw(screen._tty_fd)
        _redraw()

        while True:
            if resize_pending[0]:
                resize_pending[0] = False
                _redraw()

            rlist, _, _ = select.select([screen._tty_fd], [], [], 0.05)
            if not rlist:
                continue
            key = read_key(screen._tty_fd)

            mouse = parse_mouse(key)
            if mouse is not None:
                from .model import overlay_click_index
                action, _button, mcol, mrow = mouse
                n = len(tool_entries)
                if action == 'wheel_up':
                    lst.selected_idx = max(0, lst.selected_idx - 1)
                elif action == 'wheel_down':
                    lst.selected_idx = min(max(0, n - 1), lst.selected_idx + 1)
                elif action == 'press':
                    idx = overlay_click_index(screen._rows, screen._cols, n,
                                              lst.selected_idx, mrow, mcol)
                    if idx is not None:
                        lst.selected_idx = idx
                _redraw()
                continue

            _inner_w, visible_n, _start_r, _start_c = _box_geometry(screen._rows, screen._cols)
            if lst.handle_key(key, page=visible_n // 2):
                break
            _redraw()

    finally:
        termios.tcsetattr(screen._tty_fd, termios.TCSADRAIN, old_attrs)
        signal.signal(signal.SIGWINCH, orig_handler)
        sys.stdout.write(f'{ALT_EXIT}{CUR_HIDE}')
        sys.stdout.flush()

    return lst.checked
