"""Generic tree overlay: a tree(1)-style hierarchy on the left, a live preview
on the right. Reusable chrome for any parent/child data - the :agents view is
the first caller, future hierarchical pickers can reuse it as-is.

The drawing is shared with the model/session picker (model._draw_model_overlay
renders the list+preview box); this module only flattens a node forest into
tree rows with ├──/└── connectors, keeps the structure stable across refreshes,
and drives the modal navigation (j/k, gg/G, ^U/^D, /search, Enter, an optional
action key, ESC).

A node is a dict the caller owns; it MUST carry 'id' (unique, hashable) and
'parent' (the id of its parent, or None for a root). Everything else - the row
label, its color, the preview - the caller supplies through callbacks, so this
module stays ignorant of what a node means. fetch_fn() is re-called every
~0.5s; a node that drops out of the fetch is KEPT (marked node['present'] =
False) so the tree never loses an entry while the view is open - the only thing
that moves between refreshes is the cursor.
"""

import sys
import time

from ..ansi import (
    ALT_ENTER,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_BOLD_AZURE,
    KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_K, KEY_CTRL_U, KEY_UP, KEY_DOWN,
    KEY_PGUP, KEY_PGDN,
)
from ..input import parse_mouse
from .frame import OverlayFrame
from .model import _draw_model_overlay, _filter_and_sort, overlay_click_index
from ..togglelist import _find_matches, _handle_search_key

_REFRESH = 0.5   # seconds between data refreshes while the view is open


def _flatten_tree(nodes):
    """nodes in insertion order -> [(node, prefix), ...] in tree(1) order: each
    root, then its descendants depth-first, with ├──/└──/│ connectors. sibling
    order is the insertion order. a node whose parent id is absent is treated as
    a root, so an orphaned subtree still shows."""
    by_id = {}
    order = []
    for node in nodes:
        by_id[node['id']] = node
        order.append(node['id'])

    children = {}
    roots = []
    for nid in order:
        parent = by_id[nid].get('parent')
        if parent is not None and parent in by_id:
            children.setdefault(parent, []).append(nid)
            continue
        roots.append(nid)

    rows = []

    def walk(nid, child_prefix, connector):
        rows.append((by_id[nid], connector))
        kids = children.get(nid, [])
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            branch = '├── '
            cont = '│   '
            if last:
                branch = '└── '
                cont = '    '
            walk(kid, child_prefix + cont, child_prefix + branch)

    for nid in roots:
        walk(nid, '', '')
    return rows


def _flatten_history(nodes):
    """like _flatten_tree, but a node's FIRST child continues the trunk inline
    at the same indentation (a linear conversation renders flat), and only the
    extra children - the forks - branch off with ├──/└── connectors. the forks
    are drawn before the trunk continues, so a fork reads as a side branch off
    the main line."""
    by_id = {}
    order = []
    for node in nodes:
        by_id[node['id']] = node
        order.append(node['id'])

    children = {}
    roots = []
    for nid in order:
        parent = by_id[nid].get('parent')
        if parent is not None and parent in by_id:
            children.setdefault(parent, []).append(nid)
            continue
        roots.append(nid)

    rows = []

    def walk(nid, connector, cont):
        rows.append((by_id[nid], connector))
        kids = children.get(nid, [])
        if not kids:
            return
        trunk = kids[0]
        forks = kids[1:]
        for i, kid in enumerate(forks):
            last = i == len(forks) - 1
            branch = cont + '├── '
            child_cont = cont + '│   '
            if last:
                branch = cont + '└── '
                child_cont = cont + '    '
            walk(kid, branch, child_cont)
        walk(trunk, cont, cont)

    for nid in roots:
        walk(nid, '', '')
    return rows


def prompt_tree_overlay(screen,
                        fetch_fn,
                        *,
                        label_fn,
                        preview_fn,
                        color_fn=None,
                        is_self_fn=None,
                        action_fn=None,
                        flatten_fn=_flatten_tree,
                        filter_fn=None,
                        preview_from_top=False,
                        title="tree",
                        hints='  j/k /:search ↵:open ESC:cancel'):
    """live tree picker. fetch_fn() -> list of node dicts (each with 'id' and
    'parent'); it is re-called every ~0.5s. label_fn(node) -> row text (no
    connectors); preview_fn(node, width, max_lines) -> preview lines; color_fn
    (node) -> an SGR color for the row (optional); is_self_fn(node) -> True for
    the 'you are here' node, drawn bold/azure (optional); action_fn(id) runs on
    Ctrl-K (optional). returns the selected node id on Enter, or None on cancel.
    a node that disappears from a later fetch is retained with node['present'] =
    False so the tree stays stable - render it accordingly in your callbacks.
    flatten_fn picks the row layout: the default nests every child; _flatten_history
    keeps the first child on the trunk so a linear chain renders flat.

    filter_fn(pattern) -> the ids to show, best first: when given, '/' filters
    instead of jumping - the rows narrow to what it returns as the pattern is
    typed (the caller decides what a pattern matches: the sessions view
    searches every message, not just the row labels). Enter keeps the filter
    and returns to navigation, Esc (or an emptied pattern) lifts it.
    preview_from_top reads the preview from its first line: preview_fn then
    returns the head, and PgDn/PgUp page down from it and back."""
    nodes_by_id = {}
    order = []

    def _merge(fetched):
        present = set()
        for node in (fetched or []):
            nid = node['id']
            present.add(nid)
            node['present'] = True
            if nid not in nodes_by_id:
                order.append(nid)
            nodes_by_id[nid] = node
        for nid in order:
            if nid in present: continue
            nodes_by_id[nid]['present'] = False

    _merge(fetch_fn())
    if not nodes_by_id:
        return None

    flat = []          # [(node, connector), ...] in tree order, rebuilt each refresh
    labels = []        # display strings parallel to flat (connector + label)
    filtered = []
    selected_idx = 0
    preview_scroll = 0   # lines the preview pane is paged up from its tail
    prev_lines = {}
    first_draw = [True]

    search_mode = False
    search_direction = 1
    search_pattern = ''
    search_buf = []
    search_matches = []
    search_match_idx = -1
    pre_search_idx = 0
    prev_key = ''

    active_filter = ''   # the pattern the rows are currently narrowed by
    allowed = None       # ids filter_fn returned for it, best first; None = all

    def _rebuild_rows():
        nonlocal flat, labels, filtered
        ordered = []
        if allowed is None:
            for nid in order:
                ordered.append(nodes_by_id[nid])
        else:
            for nid in allowed:
                node = nodes_by_id.get(nid)
                if node is None: continue
                ordered.append(node)
        flat = flatten_fn(ordered)
        labels = []
        for node, connector in flat:
            labels.append(f"{connector}{label_fn(node)}")
        filtered = _filter_and_sort(labels, '')

    def _selected_id():
        if flat and 0 <= selected_idx < len(flat):
            return flat[selected_idx][0]['id']
        return None

    def _row_colors():
        colors = []
        for node, _connector in flat:
            color = ''
            if color_fn is not None:
                color = color_fn(node) or ''
            if is_self_fn is not None and is_self_fn(node):
                color = SGR_BOLD_AZURE
            colors.append(color)
        return colors

    def preview_render(_label, width, max_lines):
        if not (flat and 0 <= selected_idx < len(flat)):
            return []
        return preview_fn(flat[selected_idx][0], width, max_lines)

    def _apply_filter(pattern):
        """narrow the rows to filter_fn(pattern) - a no-op until the pattern
        changes, so navigating a filtered list never re-runs the search. a
        new pattern puts the cursor on the best match; lifting the filter
        leaves it on the row it was on."""
        nonlocal active_filter, allowed, selected_idx, preview_scroll
        nonlocal search_matches, search_match_idx
        search_matches = []          # '/' filters here: no n/N jump list
        search_match_idx = -1
        if pattern == active_filter:
            selected_idx = min(selected_idx, max(0, len(filtered) - 1))
            return
        keep_id = _selected_id()
        active_filter = pattern
        allowed = None
        if pattern:
            allowed = list(filter_fn(pattern) or [])
        _rebuild_rows()
        selected_idx = 0
        if not pattern and keep_id is not None:
            for i, (node, _connector) in enumerate(flat):
                if node['id'] != keep_id: continue
                selected_idx = i
                break
        preview_scroll = 0

    def _refresh_data(purge=False):
        nonlocal selected_idx, search_matches, search_match_idx, allowed
        keep_id = _selected_id()
        if purge:
            order.clear()
            nodes_by_id.clear()
        _merge(fetch_fn())
        if filter_fn is not None and active_filter:
            # the data moved under the filter (a row deleted, more rows
            # loaded): ask again so the narrowed list keeps up.
            allowed = list(filter_fn(active_filter) or [])
        _rebuild_rows()
        if keep_id is not None:
            for i, (node, _connector) in enumerate(flat):
                if node['id'] != keep_id: continue
                selected_idx = i
                break
        selected_idx = min(selected_idx, max(0, len(filtered) - 1))
        if search_pattern and filter_fn is None:
            search_matches = _find_matches(filtered, search_pattern)
            search_match_idx = min(search_match_idx, len(search_matches) - 1)

    def _redraw():
        nonlocal preview_scroll
        preview_scroll = _draw_model_overlay(
            screen._rows, screen._cols,
            filtered, selected_idx, search_buf,
            len(filtered), prev_lines, first_draw[0], title,
            preview_render,
            True, search_mode, search_matches, search_match_idx,
            search_direction, search_pattern,
            hints,
            _row_colors(),
            preview_scroll=preview_scroll,
            preview_from_top=preview_from_top,
        )
        first_draw[0] = False

    _rebuild_rows()

    frame = OverlayFrame(screen)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()

    result = None
    try:
        frame.enter()
        _redraw()
        last_refresh = time.monotonic()

        while True:
            if frame.resized():
                first_draw[0] = True
                _redraw()

            now = time.monotonic()
            if now - last_refresh >= _REFRESH:
                last_refresh = now
                _refresh_data()
                _redraw()

            key = frame.read_key()
            if key is None:
                continue

            mouse = parse_mouse(key)
            if mouse is not None:
                action, _button, mcol, mrow = mouse
                n = len(filtered)
                before = selected_idx
                if action == 'wheel_up':
                    selected_idx = max(0, selected_idx - 1)
                elif action == 'wheel_down':
                    selected_idx = min(max(0, n - 1), selected_idx + 1)
                elif action == 'press':
                    idx = overlay_click_index(screen._rows, screen._cols, n,
                                              selected_idx, mrow, mcol)
                    if idx is not None:
                        selected_idx = idx
                if selected_idx != before:
                    preview_scroll = 0
                _redraw()
                continue

            if search_mode:
                before = selected_idx
                search_mode, selected_idx, search_pattern, search_buf, \
                    search_matches, search_match_idx = _handle_search_key(
                        key, filtered, search_mode, selected_idx,
                        search_pattern, search_buf, search_matches,
                        search_match_idx, pre_search_idx, search_direction,
                    )
                if filter_fn is not None:
                    typed = search_pattern
                    if search_mode:
                        typed = ''.join(search_buf)
                    _apply_filter(typed)
                if selected_idx != before:
                    preview_scroll = 0
                _redraw()
                continue

            if key == KEY_ESC or key == KEY_CTRL_C:
                break

            if key in KEY_ENTER:
                result = _selected_id()
                break

            if key == KEY_CTRL_K:
                if action_fn is not None:
                    nid = _selected_id()
                    if nid is not None:
                        action_fn(nid)
                        _refresh_data(purge=True)
                _redraw()
                continue

            # PgUp/PgDn page the preview pane (not the list): scroll up from
            # the tail and back down. clamping happens in the renderer.
            if key in (KEY_PGUP, KEY_PGDN):
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                page = max(1, vis - 1)
                # scroll counts away from the pane's anchor: up from a tail,
                # down from a head - so the keys swap with preview_from_top.
                away = KEY_PGUP
                if preview_from_top:
                    away = KEY_PGDN
                if key == away:
                    preview_scroll += page
                else:
                    preview_scroll = max(0, preview_scroll - page)
                prev_key = key
                _redraw()
                continue

            n = len(filtered)
            before = selected_idx
            if key in (KEY_UP, 'k'):
                selected_idx = max(0, selected_idx - 1)
            elif key in (KEY_DOWN, 'j'):
                selected_idx = min(max(0, n - 1), selected_idx + 1)
            elif key in ('/', '?'):
                search_mode = True
                search_direction = 1
                if key == '?':
                    search_direction = -1
                search_buf = []
                search_pattern = ''
                search_matches = []
                search_match_idx = -1
                pre_search_idx = selected_idx
            elif key == 'n' and search_matches:
                search_match_idx = (search_match_idx + search_direction) % len(search_matches)
                selected_idx = search_matches[search_match_idx]
            elif key == 'N' and search_matches:
                search_match_idx = (search_match_idx - search_direction) % len(search_matches)
                selected_idx = search_matches[search_match_idx]
            elif key == 'G':
                selected_idx = max(0, n - 1)
            elif key == 'g' and prev_key == 'g':
                selected_idx = 0
            elif key == KEY_CTRL_U:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = max(0, selected_idx - max(1, vis // 2))
            elif key == KEY_CTRL_D:
                overhead = 4
                vis = max(5, int(screen._rows * 0.95)) - overhead
                selected_idx = min(max(0, n - 1), selected_idx + max(1, vis // 2))
            if selected_idx != before:
                preview_scroll = 0
            prev_key = key
            _redraw()
            continue

    finally:
        frame.leave()
        # stay in the alternate screen: the main TUI lives there too, so
        # leaving it would flash the terminal's primary buffer (the shell's
        # scrollback) until the screen's restore repaints the session.
        sys.stdout.write(CUR_HIDE)
        sys.stdout.flush()

    return result
