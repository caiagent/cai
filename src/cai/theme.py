"""theme: colour themes for the TUI, applied by remapping the terminal.

A theme is a JSON file - <name>.json in ~/.config/cai/themes/, or one shipped in
builtins/themes/ (a user file of the same name wins). The selected one is the
`theme` key of config.json; "default" (or no key) leaves the terminal alone.

cai's UI emits plain 16-colour and 256-colour SGR codes. Rather than rewrite
those, a theme REDEFINES what the terminal shows for them: OSC 10/11/12 set the
default text / background / cursor colours and OSC 4 sets a palette slot. So
the whole screen takes the background, text already on screen recolours at
once, and reset() (OSC 104/110/111/112) hands the terminal its own colours
back. Terminals without these sequences ignore them.

The file is written by role, not by slot:

  background / foreground / cursor   "#RRGGBB"
  ansi    black red green yellow blue magenta cyan white, and bright_<each>
  ui      accent bar_background tab_text chip_pink chip_yellow chip_green
  syntax  comment keyword string number function builtin deleted code_text
          (the markdown extension's code colours)

Every key is optional; a missing one leaves that colour as the terminal has it.
"""
import json
import os
import re
import sys

from cai import config

DEFAULT = "default"

_HEX_RE = re.compile(r'^#[0-9a-fA-F]{6}$')

_ST = '\033\\'

# OSC number for each top-level colour key.
_BASE_OSC = {"foreground": 10, "background": 11, "cursor": 12}

_ANSI_NAMES = ("black", "red", "green", "yellow", "blue", "magenta", "cyan", "white")

# palette slot for each `ansi` key: 0-7, then 8-15 for the bright ones.
_ANSI_SLOTS = {}
for _i, _name in enumerate(_ANSI_NAMES):
    _ANSI_SLOTS[_name] = _i
    _ANSI_SLOTS["bright_" + _name] = _i + 8

# the 256-colour slots the UI already emits, by the role they play there
# (see screen/ansi.py and tui/tabs.py).
_UI_SLOTS = {
    "accent": 45,           # SGR_BOLD_AZURE, status bar text, current tab
    "bar_background": 238,  # status bar, tab line, chips
    "tab_text": 250,
    "chip_pink": 213,
    "chip_yellow": 221,
    "chip_green": 42,
}

# the slots the markdown extension's code highlighting uses. several of its
# roles share a slot (number and inline code are both 215, function and link
# both 75), so one key covers each slot.
_SYNTAX_SLOTS = {
    "comment": 245,
    "keyword": 176,
    "string": 114,
    "number": 215,
    "function": 75,
    "builtin": 80,
    "deleted": 203,
    "code_text": 252,
}

_SECTIONS = {"ansi": _ANSI_SLOTS, "ui": _UI_SLOTS, "syntax": _SYNTAX_SLOTS}

# the theme the terminal currently shows.
_active = DEFAULT


def builtin_themes_dir():
    return os.path.join(os.path.dirname(__file__), "builtins", "themes")


def themes_dirs():
    """where themes are looked up, first match wins: the user's, then cai's."""
    return [config.themes_dir(), builtin_themes_dir()]


def list_themes():
    """every selectable theme name: "default" first, then the files, sorted."""
    names = set()
    for directory in themes_dirs():
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if entry.endswith(".json") and not entry.startswith("."):
                names.add(entry[:-len(".json")])
    names.discard(DEFAULT)
    return [DEFAULT] + sorted(names)


def _check_color(path, key, value):
    if not isinstance(value, str) or not _HEX_RE.match(value):
        raise ValueError(f"theme {path}: '{key}' must be a colour like \"#282A36\", got {value!r}")


def load_theme(name):
    """read and validate theme `name`. "default" is the empty theme. raises
    ValueError for an unknown name or a malformed file, naming the file and key."""
    if name == DEFAULT:
        return {}
    # the name comes from config.json, so it may be anything: only a plain
    # file name is looked up, never a path.
    if not isinstance(name, str) or not name or name != os.path.basename(name) or name.startswith("."):
        raise ValueError(f"theme must be the name of a theme file, got {name!r}")
    path = None
    for directory in themes_dirs():
        candidate = os.path.join(directory, name + ".json")
        if os.path.isfile(candidate):
            path = candidate
            break
    if path is None:
        raise ValueError(f"no theme named '{name}' in {config.themes_dir()}")
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as e:
        raise ValueError(f"theme {path} could not be read: {e}")
    except ValueError as e:
        raise ValueError(f"theme {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"theme {path} must be a JSON object")

    for key in _BASE_OSC:
        if key in data:
            _check_color(path, key, data[key])
    for section, slots in _SECTIONS.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"theme {path}: '{section}' must be an object")
        for key, value in values.items():
            if key not in slots:
                known = ", ".join(slots)
                raise ValueError(f"theme {path}: unknown key '{section}.{key}' (known: {known})")
            _check_color(path, f"{section}.{key}", value)
    return data


def usable_themes():
    """list_themes() without the ones whose file does not load."""
    names = []
    for name in list_themes():
        try:
            load_theme(name)
        except ValueError:
            continue
        names.append(name)
    return names


def _rgb(color):
    """"#282A36" -> "rgb:28/2a/36", the X11 colour spec OSC sequences take."""
    digits = color[1:].lower()
    return f"rgb:{digits[0:2]}/{digits[2:4]}/{digits[4:6]}"


def reset_sequence():
    """give the terminal back its own palette, text, background and cursor."""
    return f'\033]104{_ST}\033]110{_ST}\033]111{_ST}\033]112{_ST}'


def apply_sequence(theme):
    """the escape sequences that put a loaded theme on the terminal."""
    out = []
    for key, osc in _BASE_OSC.items():
        if key in theme:
            out.append(f'\033]{osc};{_rgb(theme[key])}{_ST}')
    for section, slots in _SECTIONS.items():
        for key, value in theme.get(section, {}).items():
            out.append(f'\033]4;{slots[key]};{_rgb(value)}{_ST}')
    return ''.join(out)


def active():
    return _active


def apply(name, out=None):
    """show theme `name` on the terminal. always resets first, so a theme that
    sets fewer colours does not keep the previous one's. raises ValueError (and
    changes nothing) when the theme cannot be loaded."""
    global _active
    theme = load_theme(name)
    if name == DEFAULT and _active == DEFAULT:
        return
    if out is None:
        out = sys.stdout
    out.write(reset_sequence() + apply_sequence(theme))
    out.flush()
    _active = name


def reset(out=None):
    """restore the terminal's own colours, if a theme changed them."""
    global _active
    if _active == DEFAULT:
        return
    if out is None:
        out = sys.stdout
    try:
        out.write(reset_sequence())
        out.flush()
    except (OSError, ValueError):
        pass
    _active = DEFAULT
