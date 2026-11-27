"""raw key reading and stateless input helpers."""

import os
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import tty

from .ansi import CUR_SHOW, ALT_EXIT, ansi_strip, PASTE_START, PASTE_END

# tag identifying a bracketed-paste event returned by read_key: (PASTE, text).
PASTE = 'paste'


def _read_byte(tty_fd):
    """read one more byte if it's already waiting, else ''. used to finish an
    escape sequence without blocking on a garbled/truncated one."""
    ready, _, _ = select.select([tty_fd], [], [], 0.05)
    if not ready:
        return ''
    return os.read(tty_fd, 1).decode('utf-8', errors='replace')


def _read_csi(tty_fd):
    """read a CSI sequence body (everything after the '\\033[') up to and
    including its final byte (0x40-0x7e). reading to the terminator - rather
    than a fixed-size slurp - keeps a variable-length mouse report whole and
    leaves a following batched event in the buffer for the next read."""
    seq = ''
    while True:
        b = _read_byte(tty_fd)
        if b == '':
            break
        seq += b
        if '\x40' <= b <= '\x7e':
            break
    return seq


def _read_paste(tty_fd):
    """slurp a bracketed-paste body (after PASTE_START) up to and excluding
    PASTE_END. reads raw bytes in chunks and decodes once, so multibyte UTF-8 in
    the paste is preserved. any input following PASTE_END in the same burst is
    dropped (a keystroke landing in the same read as a paste end is rare)."""
    data = b''
    end = PASTE_END.encode('utf-8')
    while True:
        ready, _, _ = select.select([tty_fd], [], [], 0.1)
        if not ready:
            break
        chunk = os.read(tty_fd, 4096)
        if chunk == b'':
            break
        data += chunk
        if end in data:
            data = data.split(end, 1)[0]
            break
    return data.decode('utf-8', errors='replace')


def read_key(tty_fd):
    """read one logical keypress from the given fd (handles escape sequences).

    a bracketed paste is returned as the tuple (PASTE, text) so the caller can
    insert the whole block literally instead of treating a pasted newline as
    Enter."""
    ch = os.read(tty_fd, 1).decode('utf-8', errors='replace')
    if ch != '\033':
        return ch
    nxt = _read_byte(tty_fd)
    if nxt == '':
        return ch
    if nxt == '[':
        seq = '\033[' + _read_csi(tty_fd)
        if seq == PASTE_START:
            return (PASTE, _read_paste(tty_fd))
        return seq
    if nxt == 'O':
        return '\033O' + _read_byte(tty_fd)
    return ch + nxt


def parse_mouse(seq):
    """decode an SGR mouse report (\\033[<b;x;y M|m) into
    (action, button, col, row), or None if seq isn't one. action is one of
    'wheel_up', 'wheel_down', 'press', 'drag', 'release'. col/row are 1-indexed
    terminal coordinates."""
    if not isinstance(seq, str):
        return None
    if not seq.startswith('\033[<'):
        return None
    if not (seq.endswith('M') or seq.endswith('m')):
        return None
    final = seq[-1]
    parts = seq[3:-1].split(';')
    if len(parts) != 3:
        return None
    try:
        b = int(parts[0])
        col = int(parts[1])
        row = int(parts[2])
    except ValueError:
        return None

    if b & 64:
        action = 'wheel_up'
        if b & 1:
            action = 'wheel_down'
        return action, b & 3, col, row

    button = b & 3
    if final == 'm':
        return 'release', button, col, row
    if b & 32:
        return 'drag', button, col, row
    return 'press', button, col, row


def input_pos_from_click(input_buf, vrow, tcol, prompt_prefix, cont_prefix, cols):
    """map a click in the input area to an index into input_buf. vrow is the
    0-based visual row within the whole prompt (the caller adds the rows
    scrolled off above the input area), tcol the 0-based terminal column.
    mirrors layout.render_input's wrapping; clamps out-of-range clicks to the
    nearest valid position."""
    cols = max(1, cols)
    buf_str = ''.join(input_buf)
    lines = buf_str.split('\n')

    base = 0
    row_in_line = vrow
    target = -1
    for i, line in enumerate(lines):
        prefix = prompt_prefix
        if i > 0:
            prefix = cont_prefix
        vrows = (len(prefix) + len(line)) // cols + 1
        if row_in_line < vrows:
            target = i
            break
        row_in_line -= vrows
        base += len(line) + 1

    if target == -1:
        return len(input_buf)

    prefix = prompt_prefix
    if target > 0:
        prefix = cont_prefix
    char_in_line = row_in_line * cols + tcol - len(prefix)
    if char_in_line < 0:
        char_in_line = 0
    if char_in_line > len(lines[target]):
        char_in_line = len(lines[target])
    return base + char_in_line


def delete_word_before(buf, pos):
    """Ctrl-W / Ctrl-Backspace: delete word before cursor. returns (buf, pos)."""
    buf = list(buf)
    while pos > 0 and buf[pos - 1] in ' \t':
        del buf[pos - 1]
        pos -= 1
    while pos > 0 and buf[pos - 1] not in ' \t\n':
        del buf[pos - 1]
        pos -= 1
    return buf, pos


def history_navigate(direction, history, history_idx, input_buf, cursor_pos):
    """navigate command history.
    returns (new_history_idx, new_buf, new_cursor_pos)."""
    new_idx = history_idx + direction

    if direction > 0 and new_idx < len(history):
        new_buf = list(history[new_idx])
        return new_idx, new_buf, len(new_buf)

    if direction < 0 and history_idx > 0:
        new_buf = list(history[history_idx - 1])
        return history_idx - 1, new_buf, len(new_buf)

    if direction < 0 and history_idx == 0:
        return -1, [], 0

    return history_idx, input_buf, cursor_pos


# what cai knows about an editor beyond "run it on a path and wait for it to
# exit" - the contract $EDITOR has everywhere (git, ranger, crontab -e). the
# keys are program basenames; an editor not listed just gets the path.
#   wait:     flags of which one must be present for the command to block
#             until the file is closed (a GUI editor returns at once without
#             it); the first is added when the user's command has none.
#   readonly: flags that open the file read-only.
#   goto:     arguments that open `path` at row / col (both 1-based), with
#             the path among them; without it the path is simply appended.
#   window:   the editor opens in a window of its own, so the terminal has
#             nothing to show while it runs.
_VIM = {'readonly': ['-R'], 'goto': ['+{row}', '+normal! {col}|', '{path}']}
_GVIM = dict(_VIM, wait=['-f', '--nofork'], window=True)
_VSCODE = {'wait': ['--wait', '-w'], 'goto': ['--goto', '{path}:{row}:{col}'], 'window': True}
_PATH_ROW_COL = ['{path}:{row}:{col}']
_PLUS_ROW_COL = ['+{row}:{col}', '{path}']
_EDITORS = {
    'vi': _VIM, 'vim': _VIM, 'nvim': _VIM, 'view': _VIM,
    'gvim': _GVIM, 'mvim': _GVIM,
    'nano': {'readonly': ['-v'], 'goto': ['+{row},{col}', '{path}']},
    'emacs': {'goto': _PLUS_ROW_COL},
    'emacsclient': {'goto': _PLUS_ROW_COL},
    'hx': {'goto': _PATH_ROW_COL},
    'helix': {'goto': _PATH_ROW_COL},
    'kak': {'readonly': ['-ro'], 'goto': _PLUS_ROW_COL},
    'micro': {'readonly': ['-readonly', 'true'], 'goto': _PLUS_ROW_COL},
    'code': _VSCODE, 'code-insiders': _VSCODE, 'codium': _VSCODE,
    'cursor': _VSCODE, 'windsurf': _VSCODE,
    'subl': {'wait': ['--wait', '-w'], 'goto': _PATH_ROW_COL, 'window': True},
    'sublime_text': {'wait': ['--wait', '-w'], 'goto': _PATH_ROW_COL, 'window': True},
    'zed': {'wait': ['--wait', '-w'], 'goto': _PATH_ROW_COL, 'window': True},
    'mate': {'wait': ['-w', '--wait'], 'goto': ['-l', '{row}:{col}', '{path}'], 'window': True},
    'gedit': {'wait': ['--wait', '-w', '-s', '--standalone'], 'goto': _PLUS_ROW_COL, 'window': True},
    'kate': {'wait': ['--block', '-b'], 'goto': ['-l', '{row}', '-c', '{col}', '{path}'],
             'window': True},
}

# tried in order when neither $VISUAL nor $EDITOR is set.
_FALLBACK_EDITORS = ('nvim', 'vim', 'vi', 'nano')


def _editor_command():
    """the user's editor as an argv prefix: $VISUAL, then $EDITOR (the order
    git and most tools use), split like a shell would so 'code --wait' or a
    quoted path works; without either, the first fallback editor installed."""
    for variable in ('VISUAL', 'EDITOR'):
        value = os.environ.get(variable, '').strip()
        if not value: continue
        try:
            argv = shlex.split(value)
        except ValueError:
            argv = value.split()
        if argv:
            return argv
    for name in _FALLBACK_EDITORS:
        if shutil.which(name):
            return [name]
    return ['vi']


def _editor_traits(argv):
    """the _EDITORS entry for the program an editor argv runs ({} for one
    cai does not know). matched on the basename, without a Windows '.exe'."""
    name = os.path.basename(argv[0]).lower()
    for ending in ('.exe', '.cmd', '.bat'):
        if name.endswith(ending):
            name = name[:-len(ending)]
            break
    return _EDITORS.get(name, {})


def editor_argv(path, *, readonly=False, row=None, col=None):
    """build the command line that opens `path` in the user's editor and
    blocks until it is closed. the editor's own flags for waiting, read-only
    and cursor position (row / col, 1-based) are added when cai knows the
    editor (see _EDITORS); any other editor just gets the path."""
    argv = _editor_command()
    traits = _editor_traits(argv)
    wait = traits.get('wait')
    if wait:
        given = False
        for flag in wait:
            if flag in argv[1:]:
                given = True
                break
        if not given:
            argv.append(wait[0])
    if readonly:
        argv.extend(traits.get('readonly', []))
    goto = traits.get('goto')
    if row is None or not goto:
        argv.append(path)
        return argv
    if col is None:
        col = 1
    for part in goto:
        argv.append(part.format(path=path, row=row, col=col))
    return argv


def run_editor(path, *, readonly=False, row=None, col=None):
    """open `path` in the user's editor and wait for it to be closed; the
    file on disk is the result. the terminal must be in cooked mode. an
    editor with a window of its own leaves the terminal empty meanwhile, so
    a line says what is being waited for; one that cannot be started is
    reported and held on screen until Enter. returns whether the editor ran."""
    argv = editor_argv(path, readonly=readonly, row=row, col=col)
    if _editor_traits(argv).get('window'):
        sys.stdout.write(f'\r\nwaiting for {argv[0]} - close the file there to come back...\r\n')
        sys.stdout.flush()
    try:
        subprocess.run(argv)
    except OSError as e:
        sys.stdout.write(f'\r\ncould not run the editor {argv[0]!r}: {e}\r\n'
                         'set $VISUAL or $EDITOR to your editor\'s command. press Enter...')
        sys.stdout.flush()
        try:
            os.read(sys.stdin.fileno(), 1024)
        except OSError:
            pass
        return False
    return True


def open_in_editor(tty_fd, cooked_attrs, buf):
    """open buf content in the editor; return updated character list."""
    content = ''.join(buf)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        sys.stdout.write(CUR_SHOW)
        sys.stdout.flush()
        run_editor(tmp)
        tty.setraw(tty_fd)
        with open(tmp, 'r') as f:
            new_content = f.read().rstrip('\n')
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return list(new_content)


def open_buffer_in_editor(tty_fd, cooked_attrs, lines, cursor_row, cursor_col):
    """open content buffer in the editor with cursor at the given position."""
    content = '\n'.join(ansi_strip(line) for line in lines)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, cooked_attrs)
        sys.stdout.write(ALT_EXIT + CUR_SHOW)
        sys.stdout.flush()
        # cursor_row is 0-based, vim is 1-based
        run_editor(tmp, row=cursor_row + 1, col=cursor_col + 1)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
