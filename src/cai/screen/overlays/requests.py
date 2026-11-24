"""Requests overlay: the HTTP requests an agent sent to the model server.

A read-only cousin of the messages overlay, with the same feel: one folded row
per request (when, status, what, how long, how big), Tab unfolds a digest of
what was sent and what came back, Enter opens the whole exchange as JSON in
the editor. It refreshes itself while open, so a request in flight - or a
string of retries - shows up as it happens.

The caller owns the data: fetch_fn() returns the summaries (see
cai.request_records.RequestRecord.summary), detail_fn(id) the full record of
one request. This module is pure presentation.

Keys: j/k move, gg/G jump, Ctrl-U/D half-page, PgUp/PgDn page, Tab (or za)
fold, zM/zR fold/unfold all, /search (n/N to cycle), Enter opens in the
editor, ESC/q closes.
"""
import json
import sys
import time

from ..ansi import (
    ALT_ENTER,
    CUR_HIDE,
    ERASE_SCREEN,
    SGR_RESET, SGR_REVERSE, SGR_GREEN, SGR_RED, SGR_YELLOW, SGR_DIM_GRAY,
    SYNC_START, SYNC_END,
    ansi_pad, ansi_sanitize,
    cur_move,
    display_truncate, display_width,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN, KEY_PGUP, KEY_PGDN,
)
from ..input import parse_mouse
from .frame import OverlayFrame

# how often the open overlay asks the agent for news - a socket round trip.
_POLL_INTERVAL = 0.5

# the states of a request that will not change anymore (cai.request_records).
_ENDED = ('done', 'failed', 'aborted')

# how much of a digest's long parts the unfolded row shows; Enter has it all.
_DIGEST_MESSAGES = 6
_DIGEST_TEXT_LINES = 12
_DIGEST_LINE_CHARS = 200


# --------------------------------------------------------------------------
# text: turning a record into the lines the overlay shows
# --------------------------------------------------------------------------

def _seconds(value):
    if value is None:
        return ''
    return f'{value:.2f}s'


def _tokens(value):
    if value is None:
        return '?'
    if value >= 1000:
        return f'{value / 1000:.1f}k'
    return str(value)


def status_label(summary):
    """the short status column of a row: the HTTP status once there is one,
    else what the request is doing."""
    if summary.get('status') is not None:
        return str(summary['status'])
    if summary.get('state') == 'failed':
        return 'ERR'            # no response at all: a network error
    if summary.get('state') == 'aborted':
        return 'CUT'
    return '...'                # still waiting for the server


def status_color(summary):
    state = summary.get('state')
    if state in ('pending', 'receiving'):
        return SGR_YELLOW
    if state == 'aborted':
        return SGR_DIM_GRAY
    if state == 'failed':
        return SGR_RED
    return SGR_GREEN


def summary_line(summary):
    """the one-line overview of a request, without its status column."""
    clock = time.strftime('%H:%M:%S', time.localtime(summary.get('started') or 0))
    parts = [clock, f"{summary.get('method')} {summary.get('path')}"]
    if summary.get('model'):
        parts.append(str(summary['model']))

    state = summary.get('state')
    if state == 'pending':
        parts.append('waiting for the server')
    elif state == 'receiving':
        parts.append('receiving')
    elif summary.get('duration') is not None:
        parts.append(_seconds(summary['duration']))
    if state == 'aborted':
        parts.append('interrupted')

    parts.append(f"{summary.get('messages', 0)} msgs")
    if summary.get('prompt_tokens') is not None or summary.get('completion_tokens') is not None:
        parts.append(f"{_tokens(summary.get('prompt_tokens'))}"
                     f"→{_tokens(summary.get('completion_tokens'))} tok")
    if summary.get('retry_after'):
        parts.append(f"retry-after {summary['retry_after']}")
    if state == 'failed' and summary.get('error'):
        first = str(summary['error']).strip().split('\n', 1)[0]
        parts.append(first[:80])
    return '  '.join(parts)


def _one_line(value):
    """any message content squeezed onto one short line."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return ' '.join(value.split())[:_DIGEST_LINE_CHARS]


def _text_block(title, text):
    """a titled block of a long text, cut to its first lines."""
    lines = [f'{title}:']
    body = text.split('\n')
    for line in body[:_DIGEST_TEXT_LINES]:
        lines.append(f'  {line}')
    if len(body) > _DIGEST_TEXT_LINES:
        lines.append(f'  ... {len(body) - _DIGEST_TEXT_LINES} more lines (Enter for all)')
    return lines


def _headers_block(title, headers):
    lines = [f'{title}:']
    if not headers:
        lines.append('  (none)')
    for name in headers:
        lines.append(f'  {name}: {headers[name]}')
    return lines


def detail_lines(detail):
    """the digest an unfolded row shows: the request and the response, with
    the long parts (the conversation, the answer) cut short."""
    request = detail.get('request') or {}
    response = detail.get('response') or {}
    body = request.get('body') or {}

    lines = [f"{request.get('method')} {request.get('url')}"]
    lines.extend(_headers_block('request headers', request.get('headers')))

    params = []
    for key in body:
        if key in ('messages', 'tools'): continue
        params.append(f'{key}={json.dumps(body[key], ensure_ascii=False, default=str)}')
    lines.append('request parameters:')
    lines.append('  ' + '  '.join(params))

    tools = body.get('tools') or []
    if tools:
        names = []
        for tool in tools:
            names.append(str((tool.get('function') or {}).get('name')))
        lines.append(f'tools ({len(tools)}): ' + ', '.join(names))

    messages = body.get('messages') or []
    lines.append(f'messages ({len(messages)}):')
    skipped = max(0, len(messages) - _DIGEST_MESSAGES)
    if skipped:
        lines.append(f'  ... {skipped} earlier (Enter for all)')
    for message in messages[skipped:]:
        text = _one_line(message.get('content') or '')
        calls = message.get('tool_calls') or []
        if calls:
            text = f'[{len(calls)} tool call(s)] {text}'
        lines.append(f"  {str(message.get('role')):<9} {text}")

    lines.append('')
    if response.get('status') is None:
        lines.append(f"response: none ({detail.get('state')})")
    else:
        timing = [f"first byte {_seconds(response.get('first_byte'))}"]
        if response.get('duration') is not None:
            timing.append(f"total {_seconds(response['duration'])}")
        if response.get('chunks'):
            timing.append(f"{response['chunks']} chunks")
        lines.append(f"response: {response['status']} ({detail.get('state')}; "
                     + ', '.join(timing) + ')')
        lines.extend(_headers_block('response headers', response.get('headers')))

    if response.get('error'):
        lines.extend(_text_block('error', str(response['error'])))
    if response.get('usage'):
        lines.append('usage: ' + json.dumps(response['usage'], ensure_ascii=False, default=str))
    if response.get('reasoning'):
        lines.extend(_text_block('reasoning', response['reasoning']))
    if response.get('content'):
        lines.extend(_text_block('content', response['content']))
    for call in response.get('tool_calls') or []:
        function = call.get('function') or {}
        lines.append(f"tool call: {function.get('name')}({_one_line(function.get('arguments') or '')})")
    return lines


def editor_text(detail):
    """the whole exchange as pretty JSON - what Enter opens in the editor."""
    return json.dumps(detail, indent=2, ensure_ascii=False, default=str)


def _wrap(line, width):
    """one text line cut into pieces of at most `width` display columns."""
    line = ansi_sanitize(line)
    if not line:
        return ['']
    pieces = []
    while display_width(line) > width:
        head = display_truncate(line, width)
        if not head:
            head = line[0]
        pieces.append(head)
        line = line[len(head):]
    pieces.append(line)
    return pieces


# --------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------

class _View:
    """the overlay's state: the summaries on show, which one the cursor is on,
    which are unfolded, and the details fetched for those."""

    def __init__(self, fetch_fn, detail_fn):
        self.fetch_fn = fetch_fn
        self.detail_fn = detail_fn
        self.summaries = []
        self.selected = 0
        self.scroll = 0
        self.unfolded = set()       # request ids
        self.details = {}           # request id -> detail dict
        self.search_typing = None   # the pattern being typed, or None
        self.search = ''
        self.flash = ''

    def selected_id(self):
        if not self.summaries:
            return None
        return self.summaries[self.selected]['id']

    def refresh(self):
        """pull the summaries again, keeping the cursor on the same request -
        or on the newest one when it was already there - and re-read the
        detail of every unfolded request that could still change."""
        keep = self.selected_id()
        was_last = self.selected >= len(self.summaries) - 1
        self.summaries = self.fetch_fn() or []

        known = set()
        for summary in self.summaries:
            known.add(summary['id'])
        self.unfolded &= known
        for rid in list(self.details):
            if rid not in known:
                del self.details[rid]

        self.selected = max(0, len(self.summaries) - 1)
        if not was_last:
            for i, summary in enumerate(self.summaries):
                if summary['id'] != keep: continue
                self.selected = i
                break

        for summary in self.summaries:
            rid = summary['id']
            if rid not in self.unfolded: continue
            cached = self.details.get(rid)
            if cached is not None and cached.get('state') in _ENDED: continue
            self.load_detail(rid)

    def load_detail(self, rid):
        detail = self.detail_fn(rid)
        if detail is None:
            self.details.pop(rid, None)
            return None
        self.details[rid] = detail
        return detail

    def toggle(self, rid):
        if rid in self.unfolded:
            self.unfolded.discard(rid)
            return
        if self.load_detail(rid) is None:
            self.flash = 'that request is no longer remembered'
            return
        self.unfolded.add(rid)

    def fold_all(self):
        self.unfolded.clear()

    def unfold_all(self):
        for summary in self.summaries:
            if self.load_detail(summary['id']) is None: continue
            self.unfolded.add(summary['id'])

    def matches(self):
        """indices of the requests whose row (or unfolded digest) contains the
        search pattern."""
        if not self.search:
            return []
        needle = self.search.lower()
        found = []
        for i, summary in enumerate(self.summaries):
            text = f'{status_label(summary)} {summary_line(summary)}'
            detail = self.details.get(summary['id'])
            if detail is not None and summary['id'] in self.unfolded:
                text += '\n' + '\n'.join(detail_lines(detail))
            if needle in text.lower():
                found.append(i)
        return found

    def jump_to_match(self, step):
        found = self.matches()
        if not found:
            self.flash = f'no match: {self.search}'
            return
        if step > 0:
            for i in found:
                if i > self.selected:
                    self.selected = i
                    return
            self.selected = found[0]
            return
        for i in reversed(found):
            if i < self.selected:
                self.selected = i
                return
        self.selected = found[-1]


def _build_rows(view, text_w):
    """flatten the view into display rows: (request index, kind, text), one
    'header' per request and, when unfolded, one 'body' per wrapped line."""
    rows = []
    for i, summary in enumerate(view.summaries):
        rid = summary['id']
        glyph = '▶'
        if rid in view.unfolded:
            glyph = '▼'
        rows.append((i, 'header', f'{glyph} {summary_line(summary)}'))
        if rid not in view.unfolded:
            continue
        detail = view.details.get(rid)
        if detail is None:
            continue
        for line in detail_lines(detail):
            for piece in _wrap(line, max(4, text_w - 2)):
                rows.append((i, 'body', piece))
    return rows


def _status_text(view, total):
    if view.search_typing is not None:
        return f' /{view.search_typing}', True
    if view.flash:
        return f' {view.flash}', True
    text = ' 0/0'
    if total:
        text = f' {view.selected + 1}/{total}'
    if view.search:
        text += f'   /{view.search} [{len(view.matches())}]'
    text += '  Tab:fold  ↵:open in editor  /:search  ESC:close'
    return text, False


def _draw(view, title, rows, cols):
    idx_w = 3
    status_w = 4
    prefix_w = 1 + idx_w + 1 + status_w + 1

    overhead = 4
    box_h = max(overhead + 1, int(rows * 0.95))
    visible_n = box_h - overhead
    inner_w = max(prefix_w + 10, int(cols * 0.95) - 2)
    box_w = inner_w + 2
    start_r = max(1, (rows - box_h) // 2 + 1)
    start_c = max(1, (cols - box_w) // 2 + 1)

    all_rows = _build_rows(view, inner_w - prefix_w)

    # keep the selected request's header on screen and, when it is unfolded
    # and fits, its whole digest under it.
    first = 0
    last = 0
    for n, row in enumerate(all_rows):
        if row[0] != view.selected: continue
        if row[1] == 'header':
            first = n
        last = n
    scroll = view.scroll
    scroll = max(scroll, last - visible_n + 1)
    scroll = min(scroll, first)
    scroll = max(0, min(scroll, max(0, len(all_rows) - visible_n)))
    view.scroll = scroll

    found = set(view.matches())
    heading = f'  {title}  '[:inner_w]
    pad_l = (inner_w - len(heading)) // 2
    pad_r = inner_w - len(heading) - pad_l

    out = [SYNC_START, CUR_HIDE]
    out.append(f"{cur_move(start_r, start_c)}┌{'─' * pad_l}{heading}{'─' * pad_r}┐")
    for line_no in range(visible_n):
        cell = ' ' * inner_w
        src = line_no + scroll
        if src < len(all_rows):
            i, kind, text = all_rows[src]
            summary = view.summaries[i]
            if kind == 'header':
                number = str(summary['id']).rjust(idx_w)
                status = status_label(summary).ljust(status_w)
                cell = ansi_pad(f' {number} {status} {text}', inner_w)
                if i == view.selected:
                    cell = f'{SGR_REVERSE}{cell}{SGR_RESET}'
                elif i in found:
                    cell = f'{SGR_YELLOW}{cell}{SGR_RESET}'
                else:
                    cell = f'{status_color(summary)}{cell}{SGR_RESET}'
            else:
                cell = ansi_pad(' ' * (prefix_w + 2) + text, inner_w)
        out.append(f'{cur_move(start_r + 1 + line_no, start_c)}│{cell}│')
    out.append(f"{cur_move(start_r + 1 + visible_n, start_c)}├{'─' * inner_w}┤")
    status, reverse = _status_text(view, len(view.summaries))
    status = ansi_pad(status, inner_w)
    if reverse:
        status = f'{SGR_REVERSE}{status}{SGR_RESET}'
    out.append(f'{cur_move(start_r + 2 + visible_n, start_c)}│{status}│')
    out.append(f"{cur_move(start_r + 3 + visible_n, start_c)}└{'─' * inner_w}┘")
    out.append(SYNC_END)
    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def _search_key(view, key):
    """one key while a search pattern is being typed."""
    if key == KEY_ESC or key == KEY_CTRL_C:
        view.search_typing = None
        return
    if key in KEY_ENTER:
        view.search = view.search_typing
        view.search_typing = None
        if view.search:
            view.jump_to_match(1)
        return
    if key == KEY_BACKSPACE:
        view.search_typing = view.search_typing[:-1]
        return
    if len(key) == 1 and key.isprintable():
        view.search_typing += key


def _open_in_editor(view, screen, frame):
    """show the selected request's whole exchange in the editor, read-only,
    then take the terminal back for the overlay."""
    rid = view.selected_id()
    if rid is None:
        return
    detail = view.load_detail(rid)
    if detail is None:
        view.flash = 'that request is no longer remembered'
        return
    screen.view_in_editor(editor_text(detail), suffix='.json')
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()
    frame.raw()


def _nav_key(view, key, prev_key, screen, frame):
    """one key in normal mode. returns 'close' to leave the overlay."""
    if key in (KEY_ESC, 'q', KEY_CTRL_C):
        return 'close'

    total = len(view.summaries)
    if total == 0:
        return None
    last = total - 1
    page = max(1, int(screen._rows * 0.95) - 5)
    half = max(1, page // 2)
    rid = view.selected_id()

    if key in (KEY_UP, 'k'):
        view.selected = max(0, view.selected - 1)
    elif key in (KEY_DOWN, 'j'):
        view.selected = min(last, view.selected + 1)
    elif key == 'G':
        view.selected = last
    elif key == 'g' and prev_key == 'g':
        view.selected = 0
    elif key == KEY_CTRL_U:
        view.selected = max(0, view.selected - half)
    elif key == KEY_CTRL_D:
        view.selected = min(last, view.selected + half)
    elif key == KEY_PGUP:
        view.selected = max(0, view.selected - page)
    elif key == KEY_PGDN:
        view.selected = min(last, view.selected + page)
    elif key == '\t' or (key == 'a' and prev_key == 'z'):
        view.toggle(rid)
    elif key == 'M' and prev_key == 'z':
        view.fold_all()
    elif key == 'R' and prev_key == 'z':
        view.unfold_all()
    elif key == '/':
        view.search_typing = ''
    elif key == 'n' and view.search:
        view.jump_to_match(1)
    elif key == 'N' and view.search:
        view.jump_to_match(-1)
    elif key in KEY_ENTER:
        _open_in_editor(view, screen, frame)
    return None


def prompt_requests_overlay(screen, fetch_fn, detail_fn, *, title='Requests'):
    """the requests overlay. fetch_fn() returns the request summaries, oldest
    first; detail_fn(id) the full record of one (None once it is gone). opens
    on the newest request and follows new ones while the cursor stays at the
    bottom. returns None when closed."""
    view = _View(fetch_fn, detail_fn)
    view.refresh()

    frame = OverlayFrame(screen)
    sys.stdout.write(f'{ALT_ENTER}{ERASE_SCREEN}')
    sys.stdout.flush()
    prev_key = ''
    try:
        frame.enter()
        _draw(view, title, screen._rows, screen._cols)
        last_poll = time.monotonic()

        while True:
            if frame.resized():
                sys.stdout.write(ERASE_SCREEN)
                _draw(view, title, screen._rows, screen._cols)

            if time.monotonic() - last_poll >= _POLL_INTERVAL:
                last_poll = time.monotonic()
                view.refresh()
                _draw(view, title, screen._rows, screen._cols)

            key = frame.read_key()
            if key is None:
                continue

            mouse = parse_mouse(key)
            if mouse is not None:
                last = max(0, len(view.summaries) - 1)
                if mouse[0] == 'wheel_up':
                    view.selected = max(0, view.selected - 1)
                elif mouse[0] == 'wheel_down':
                    view.selected = min(last, view.selected + 1)
                _draw(view, title, screen._rows, screen._cols)
                continue

            view.flash = ''
            if view.search_typing is not None:
                _search_key(view, key)
            else:
                if _nav_key(view, key, prev_key, screen, frame) == 'close':
                    break
                prev_key = key
            _draw(view, title, screen._rows, screen._cols)
    finally:
        frame.leave()
        # stay in the alternate screen: the main TUI lives there too, so
        # leaving it would flash the terminal's primary buffer (the shell's
        # scrollback) until the screen's restore repaints the session.
        sys.stdout.write(CUR_HIDE)
        sys.stdout.flush()
    return None
