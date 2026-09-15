"""ask: the inline prompt elements. a question raised mid-run (a hook's
ctx.ui.confirm, a tool's cai.current_ui().select, ...) becomes one of these,
appended to the conversation as a block the user answers in place - the
history stays visible around it. the element never touches the terminal:
handle_key mutates it and lines() renders it; the Screen owns the segment the
lines live in, repaints it after every key, and parks the cursor at `cursor`
while insert mode is answering. once answered (or abandoned) `answered` is
True, lines() renders the question with its answer, and `result` is what the
requesting UI primitive hands back."""

from .ansi import (
    SGR_RESET, SGR_BOLD, SGR_DIM_GRAY, SGR_REVERSE,
    KEY_ENTER, KEY_CTRL_C, KEY_BACKSPACE,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_TAB,
)
from .togglelist import ToggleList


def _title(text):
    return f'{SGR_BOLD}? {text}{SGR_RESET}'


def _dim(text):
    return f'{SGR_DIM_GRAY}{text}{SGR_RESET}'


def _pick(text, selected):
    if selected:
        return f'{SGR_REVERSE}{text}{SGR_RESET}'
    return text


class ConfirmAsk:
    """yes / no. h/l/arrows/Tab move between the two, Enter takes the
    highlighted one, y/a answer yes, n/d answer no, Ctrl-C the default."""

    searching = False

    def __init__(self, title, body='', default=False):
        self.title = title
        self.body = body
        self.default = default
        self.choice = 1
        if default:
            self.choice = 0
        self.answered = False
        self.result = default
        self.cursor = (0, 0)

    def handle_key(self, key):
        if key in (KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_TAB, 'h', 'l', 'j', 'k'):
            self.choice = 1 - self.choice
            return False
        if key in ('y', 'a'):
            self.result = True
        elif key in ('n', 'd'):
            self.result = False
        elif key in KEY_ENTER:
            self.result = self.choice == 0
        elif key == KEY_CTRL_C:
            self.result = self.default
        else:
            return False
        self.answered = True
        return True

    def lines(self, width):
        out = [_title(self.title)]
        for line in self.body.splitlines():
            out.append(_dim(f'  {line}'))
        if self.answered:
            answer = 'no'
            if self.result:
                answer = 'yes'
            out.append(_dim(f'  -> {answer}'))
            return out
        yes = _pick(' yes ', self.choice == 0)
        no = _pick(' no ', self.choice == 1)
        out.append(f'  {yes}  {no}')
        col = 2
        if self.choice == 1:
            col = 2 + len(' yes ') + 2
        self.cursor = (len(out) - 1, col)
        return out


class SelectAsk:
    """pick one. j/k/arrows move, Enter picks, Ctrl-C answers None."""

    searching = False

    def __init__(self, title, options):
        self.title = title
        self.options = list(options)
        self.index = 0
        self.answered = False
        self.result = None
        self.cursor = (0, 0)

    def handle_key(self, key):
        n = len(self.options)
        if key in (KEY_UP, 'k'):
            self.index = max(0, self.index - 1)
            return False
        if key in (KEY_DOWN, 'j'):
            self.index = min(n - 1, self.index + 1)
            return False
        if key in KEY_ENTER and n > 0:
            self.result = self.options[self.index]
        elif key == KEY_CTRL_C:
            self.result = None
        else:
            return False
        self.answered = True
        return True

    def lines(self, width):
        out = [_title(self.title)]
        if self.answered:
            answer = '(cancelled)'
            if self.result is not None:
                answer = self.result
            out.append(_dim(f'  -> {answer}'))
            return out
        for i, option in enumerate(self.options):
            out.append(_pick(f'  {option} ', i == self.index))
        self.cursor = (1 + self.index, 2)
        return out


class MultiAsk:
    """pick many: a ToggleList (j/k, Space toggles, / searches), Enter
    answers with the checked options in order, Ctrl-C answers None. while a
    search is being typed `searching` is True so Esc cancels the search
    instead of leaving insert mode."""

    def __init__(self, title, options, default=()):
        self.title = title
        self.options = list(options)
        entries = [(option, '') for option in self.options]
        self.list = ToggleList(entries, default)
        self.searching = False
        self.answered = False
        self.result = None
        self.cursor = (0, 0)

    def handle_key(self, key):
        closed = self.list.handle_key(key, page=5)
        self.searching = self.list.search_mode
        if not closed:
            return False
        if key == KEY_CTRL_C:
            self.result = None
        else:
            self.result = self.list.checked_in_order()
        self.answered = True
        return True

    def lines(self, width):
        out = [_title(self.title)]
        if self.answered:
            answer = '(cancelled)'
            if self.result is not None:
                answer = ', '.join(self.result)
            if self.result == []:
                answer = '(none)'
            out.append(_dim(f'  -> {answer}'))
            return out
        longest = 0
        for option in self.options:
            longest = max(longest, len(option))
        row_width = min(width, longest + 8)
        out.extend(self.list.rows(row_width, len(self.options)))
        self.cursor = (1 + self.list.selected_idx, 2)
        if self.searching:
            out.append(f'  {self.list.status(width)}')
            self.cursor = (len(out) - 1, 2 + len(self.list.status(width)))
        return out


class TextAsk:
    """one line of text. printable keys type, Backspace deletes, Enter
    answers (an empty entry answers `default`), Ctrl-C answers None."""

    searching = False

    def __init__(self, title, default='', secret=False):
        self.title = title
        self.default = default
        self.secret = secret
        self.buf = []
        self.answered = False
        self.result = None
        self.cursor = (0, 0)

    def handle_key(self, key):
        if key in KEY_ENTER:
            text = ''.join(self.buf)
            if text == '':
                text = self.default
            self.result = text
        elif key == KEY_CTRL_C:
            self.result = None
        elif key == KEY_BACKSPACE:
            if self.buf:
                self.buf.pop()
            return False
        elif len(key) == 1 and key >= ' ':
            self.buf.append(key)
            return False
        else:
            return False
        self.answered = True
        return True

    def lines(self, width):
        out = [_title(self.title)]
        typed = ''.join(self.buf)
        if self.secret:
            typed = '*' * len(self.buf)
        if self.answered:
            answer = '(cancelled)'
            if self.result is not None:
                answer = typed
                if self.result == self.default and not self.buf:
                    answer = self.default
                if self.secret and self.result:
                    answer = '*' * len(self.result)
            out.append(_dim(f'  -> {answer}'))
            return out
        out.append(f'  > {typed}')
        self.cursor = (1, 4 + len(typed))
        return out
