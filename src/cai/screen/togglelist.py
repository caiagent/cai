"""ToggleList: the state and rows of a checkbox list - a cursor over (name,
tag) entries, a checked set, vim navigation and /-search. it never touches the
terminal: handle_key mutates the state, rows/status render it to strings, and
the caller paints them wherever it likes. the fullscreen tools/skills overlay
and the inline pick-many prompt both drive one of these."""

import re

from .ansi import (
    SGR_RESET, SGR_REVERSE, SGR_YELLOW, SGR_REVERSE_YELLOW,
    KEY_BACKSPACE, KEY_ESC, KEY_ENTER, KEY_CTRL_C,
    KEY_CTRL_D, KEY_CTRL_U, KEY_UP, KEY_DOWN,
)


def _find_matches(entries, pattern):
    """search over names (first element of each entry tuple)."""
    if not pattern:
        return []
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    return [i for i, (nm, _tag) in enumerate(entries) if rx.search(nm)]


def _nearest_fwd(matches, from_idx):
    for i, m in enumerate(matches):
        if m >= from_idx:
            return i
    return 0


def _nearest_bwd(matches, from_idx):
    for i in range(len(matches) - 1, -1, -1):
        if matches[i] <= from_idx:
            return i
    return len(matches) - 1


def _sync_cursor(selected_idx, pre_search_idx, search_direction, search_matches):
    """return (new_selected_idx, new_search_match_idx)."""
    if not search_matches:
        return pre_search_idx, -1
    if search_direction == 1:
        mi = _nearest_fwd(search_matches, pre_search_idx)
    else:
        mi = _nearest_bwd(search_matches, pre_search_idx)
    return search_matches[mi], mi


def _style_cell(cell, is_sel, is_match, color=None):
    if is_sel and is_match:
        return f'{SGR_REVERSE_YELLOW}{cell}{SGR_RESET}'
    if is_sel:
        return f'{SGR_REVERSE}{cell}{SGR_RESET}'
    if is_match:
        return f'{SGR_YELLOW}{cell}{SGR_RESET}'
    if color:
        return f'{color}{cell}{SGR_RESET}'
    return cell


def _handle_search_key(key,
                       entries,
                       search_mode,
                       selected_idx,
                       search_pattern,
                       search_buf,
                       search_matches,
                       search_match_idx,
                       pre_search_idx,
                       search_direction):
    """handle one key in search input mode. returns updated search state tuple."""
    if key in KEY_ENTER:
        return (False, selected_idx, search_pattern, search_buf,
                search_matches, search_match_idx)

    if key == KEY_ESC:
        return (False, pre_search_idx, '', [],  [], -1)

    if key == KEY_BACKSPACE:
        if search_buf:
            search_buf = search_buf[:-1]
            search_pattern = ''.join(search_buf)
            search_matches = _find_matches(entries, search_pattern)
            selected_idx, search_match_idx = _sync_cursor(
                selected_idx, pre_search_idx,
                search_direction, search_matches,
            )
        else:
            return (False, pre_search_idx, '', [], [], -1)
        return (True, selected_idx, search_pattern, search_buf,
                search_matches, search_match_idx)

    if len(key) == 1 and ord(key) >= 32:
        search_buf = search_buf + [key]
        search_pattern = ''.join(search_buf)
        search_matches = _find_matches(entries, search_pattern)
        selected_idx, search_match_idx = _sync_cursor(
            selected_idx, pre_search_idx,
            search_direction, search_matches,
        )

    return (True, selected_idx, search_pattern, search_buf,
            search_matches, search_match_idx)


class ToggleList:
    """a checkbox list: entries are (name, tag) pairs, checked the names
    ticked. keys - j/k/arrows move, Space toggles, gg/G jump, Ctrl-U/D page,
    / and ? search then n/N cycle; Enter, Esc or Ctrl-C close it."""

    def __init__(self, entries, checked):
        self.entries = list(entries)
        self.checked = set(checked)
        self.selected_idx = 0
        self.search_mode = False
        self.search_direction = 1
        self.search_buf = []
        self.search_pattern = ''
        self.search_matches = []
        self.search_match_idx = -1
        self.pre_search_idx = 0
        self.prev_key = ''

    def handle_key(self, key, page=10):
        """apply one key. page is how many rows Ctrl-U/Ctrl-D move. returns
        True when the key closed the list (Enter / Esc / Ctrl-C)."""
        if self.search_mode:
            self.search_mode, self.selected_idx, self.search_pattern, self.search_buf, \
                self.search_matches, self.search_match_idx = _handle_search_key(
                    key, self.entries, self.search_mode, self.selected_idx,
                    self.search_pattern, self.search_buf, self.search_matches,
                    self.search_match_idx, self.pre_search_idx, self.search_direction,
                )
            return False

        n = len(self.entries)
        prev_key = self.prev_key
        self.prev_key = key

        if key in (KEY_ESC, *KEY_ENTER, KEY_CTRL_C):
            return True
        if key in (KEY_UP, 'k'):
            self.selected_idx = max(0, self.selected_idx - 1)
        elif key in (KEY_DOWN, 'j'):
            self.selected_idx = min(n - 1, self.selected_idx + 1)
        elif key == ' ':
            nm = self.entries[self.selected_idx][0]
            if nm in self.checked:
                self.checked.discard(nm)
            else:
                self.checked.add(nm)
        elif key in ('/', '?'):
            self.search_mode = True
            self.search_direction = -1
            if key == '/':
                self.search_direction = 1
            self.search_buf = []
            self.search_pattern = ''
            self.search_matches = []
            self.search_match_idx = -1
            self.pre_search_idx = self.selected_idx
        elif key == 'n' and self.search_matches:
            step = self.search_direction
            self.search_match_idx = (self.search_match_idx + step) % len(self.search_matches)
            self.selected_idx = self.search_matches[self.search_match_idx]
        elif key == 'N' and self.search_matches:
            step = -self.search_direction
            self.search_match_idx = (self.search_match_idx + step) % len(self.search_matches)
            self.selected_idx = self.search_matches[self.search_match_idx]
        elif key == 'G':
            self.selected_idx = n - 1
        elif key == 'g' and prev_key == 'g':
            self.selected_idx = 0
        elif key == KEY_CTRL_U:
            self.selected_idx = max(0, self.selected_idx - max(1, page))
        elif key == KEY_CTRL_D:
            self.selected_idx = min(n - 1, self.selected_idx + max(1, page))
        return False

    def checked_count(self):
        count = 0
        for nm, _tag in self.entries:
            if nm not in self.checked: continue
            count += 1
        return count

    def checked_in_order(self):
        """the checked names in entry order."""
        names = []
        for nm, _tag in self.entries:
            if nm not in self.checked: continue
            names.append(nm)
        return names

    def rows(self, width, height):
        """exactly `height` styled rows of `width` cells, scrolled so the
        cursor row is visible; rows past the last entry are blank."""
        n = len(self.entries)
        scroll = 0
        if self.selected_idx >= height:
            scroll = self.selected_idx - height + 1
        scroll = max(0, min(scroll, n - height))

        out = []
        for i in range(height):
            ai = i + scroll
            if ai >= n:
                out.append(' ' * width)
                continue
            nm, tag = self.entries[ai]
            check = '[ ]'
            if nm in self.checked:
                check = '[x]'
            tag_text = f'[{tag}]'
            if tag == '':
                tag_text = ''
            max_nm_len = max(1, width - 6 - 2 - len(tag_text))
            nm_padded = nm[:max_nm_len].ljust(max_nm_len)
            raw_line = f'  {check} {nm_padded}  {tag_text}'
            cell = raw_line[:width].ljust(width)
            is_sel = (ai == self.selected_idx)
            is_match = bool(self.search_matches) and (ai in self.search_matches)
            out.append(_style_cell(cell, is_sel, is_match))
        return out

    def status(self, width):
        """the one-line footer: the search being typed, else the checked
        count with the last search and the key hints when they fit."""
        dir_char = '?'
        if self.search_direction == 1:
            dir_char = '/'

        if self.search_mode:
            search_text = ''.join(self.search_buf)
            if self.search_matches:
                m_info = f' [{self.search_match_idx + 1}/{len(self.search_matches)}]'
            elif search_text:
                m_info = ' [no match]'
            else:
                m_info = ''
            return f' {dir_char}{search_text}{m_info}'

        count_str = f' {self.checked_count()}/{len(self.entries)} enabled'
        if self.search_pattern:
            m_label = ''
            if self.search_matches:
                m_label = f' [{self.search_match_idx + 1}/{len(self.search_matches)}]'
            count_str += f'   {dir_char}{self.search_pattern}{m_label}'
        hints = '  j/k /:search ESC/↵:close'
        if len(count_str) + len(hints) <= width:
            count_str += hints
        return count_str
