"""Tests for cai.theme - listing and loading theme files, the escape sequences
that put one on the terminal, and config.save_optional, which records the
choice. Each test gets its own ~/.config/cai via HOME."""
import io
import json
import os

import pytest

from cai import config
from cai import theme


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(theme, "_active", theme.DEFAULT)


def _write_theme(name, data):
    os.makedirs(config.themes_dir(), exist_ok=True)
    with open(os.path.join(config.themes_dir(), name + ".json"), "w") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f)


def test_dracula_ships_and_default_is_listed_first():
    assert theme.list_themes()[0] == "default"
    assert "dracula" in theme.list_themes()
    assert theme.load_theme("dracula")["background"] == "#282A36"


_SHIPPED = sorted(entry[:-len(".json")] for entry in os.listdir(theme.builtin_themes_dir())
                  if entry.endswith(".json"))


@pytest.mark.parametrize("name", _SHIPPED)
def test_every_shipped_theme_loads_and_sets_the_background(name):
    data = theme.load_theme(name)
    assert "\033]11;" in theme.apply_sequence(data)
    assert len(data["ansi"]) == 16


def test_default_is_the_empty_theme():
    assert theme.load_theme("default") == {}
    assert theme.apply_sequence({}) == ""


def test_a_user_theme_is_listed_and_overrides_a_shipped_one():
    _write_theme("mine", {"background": "#000000"})
    _write_theme("dracula", {"background": "#111111"})
    names = theme.list_themes()
    assert names[0] == "default" and names[1:] == sorted(names[1:])
    assert "mine" in names and names.count("dracula") == 1
    assert theme.load_theme("dracula") == {"background": "#111111"}


def test_apply_sequence_sets_base_colours_and_palette_slots():
    seq = theme.apply_sequence({"background": "#282A36",
                                "foreground": "#F8F8F2",
                                "ansi": {"red": "#FF5555", "bright_black": "#6272A4"},
                                "ui": {"accent": "#BD93F9", "bar_background": "#44475A"},
                                "syntax": {"keyword": "#FF79C6"}})
    assert "\033]11;rgb:28/2a/36\033\\" in seq
    assert "\033]10;rgb:f8/f8/f2\033\\" in seq
    assert "\033]4;1;rgb:ff/55/55\033\\" in seq
    assert "\033]4;8;rgb:62/72/a4\033\\" in seq
    assert "\033]4;45;rgb:bd/93/f9\033\\" in seq
    assert "\033]4;238;rgb:44/47/5a\033\\" in seq
    assert "\033]4;176;rgb:ff/79/c6\033\\" in seq


def test_every_shipped_dracula_colour_reaches_the_terminal():
    data = theme.load_theme("dracula")
    count = 3 + len(data["ansi"]) + len(data["ui"]) + len(data["syntax"])
    assert theme.apply_sequence(data).count("\033]") == count


def test_apply_resets_first_and_reset_restores_the_terminal():
    out = io.StringIO()
    theme.apply("dracula", out)
    assert theme.active() == "dracula"
    assert out.getvalue().startswith(theme.reset_sequence())
    assert "\033]11;rgb:28/2a/36" in out.getvalue()

    out = io.StringIO()
    theme.reset(out)
    assert out.getvalue() == theme.reset_sequence()
    assert theme.active() == "default"
    # nothing to undo any more: a second reset writes nothing.
    out = io.StringIO()
    theme.reset(out)
    assert out.getvalue() == ""


def test_applying_default_untouched_writes_nothing():
    out = io.StringIO()
    theme.apply("default", out)
    assert out.getvalue() == ""


@pytest.mark.parametrize("data, needle", [
    ({"background": "282A36"}, "'background'"),
    ({"ansi": {"red": "#F55"}}, "'ansi.red'"),
    ({"ui": {"nope": "#000000"}}, "unknown key 'ui.nope'"),
    ({"syntax": "#000000"}, "'syntax' must be an object"),
    ("{not json", "not valid JSON"),
    ("[]", "must be a JSON object"),
])
def test_a_malformed_theme_is_refused_with_the_key_named(data, needle):
    _write_theme("bad", data)
    with pytest.raises(ValueError) as e:
        theme.load_theme("bad")
    assert needle in str(e.value)
    assert "bad.json" in str(e.value)


def test_a_failed_apply_changes_nothing():
    out = io.StringIO()
    with pytest.raises(ValueError):
        theme.apply("no-such-theme", out)
    assert out.getvalue() == ""
    assert theme.active() == "default"


def test_save_optional_keeps_the_other_keys():
    os.makedirs(config.config_dir(), exist_ok=True)
    with open(config.config_path(), "w") as f:
        json.dump({"base_url": "u", "model": "m"}, f)
    config.save_optional("theme", "dracula")
    with open(config.config_path()) as f:
        assert json.load(f) == {"base_url": "u", "model": "m", "theme": "dracula"}
    assert config.load_optional("theme") == "dracula"
    assert not os.path.exists(config.config_path() + ".tmp")


def test_save_optional_refuses_to_overwrite_a_broken_config():
    os.makedirs(config.config_dir(), exist_ok=True)
    with open(config.config_path(), "w") as f:
        f.write("{broken")
    with pytest.raises(ValueError):
        config.save_optional("theme", "dracula")
    with open(config.config_path()) as f:
        assert f.read() == "{broken"


def test_theme_command_is_in_the_palette_and_config_has_a_theme_row():
    from cai.tui import _PALETTE_COMMANDS
    from cai.tui.pickers import _set_theme, _theme_setting
    assert "theme" in [name for name, _help in _PALETTE_COMMANDS]
    setting = _theme_setting()
    assert setting.options[0] == "default" and "dracula" in setting.options
    assert setting.get() == "default"


def test_set_theme_applies_and_saves(capsys):
    from cai.tui.pickers import _set_theme
    assert _set_theme("dracula") is None
    assert theme.active() == "dracula"
    assert config.load_optional("theme") == "dracula"
    assert "\033]11;" in capsys.readouterr().out
    assert "no theme named" in _set_theme("no-such-theme")
    assert theme.active() == "dracula"


@pytest.mark.parametrize("name", [5, None, ["dracula"], "", "../dracula", "sub/x", ".hidden"])
def test_a_theme_name_that_is_not_a_plain_name_is_refused(name):
    with pytest.raises(ValueError):
        theme.load_theme(name)
    out = io.StringIO()
    with pytest.raises(ValueError):
        theme.apply(name, out)
    assert out.getvalue() == ""


def test_usable_themes_leaves_out_a_broken_file():
    _write_theme("bad", {"background": "nope"})
    assert "bad" in theme.list_themes()
    assert "bad" not in theme.usable_themes()
    assert "dracula" in theme.usable_themes()


def test_set_theme_keeps_the_shown_theme_when_it_cannot_save(capsys):
    from cai.tui.pickers import _set_theme
    os.makedirs(config.config_dir(), exist_ok=True)
    with open(config.config_path(), "w") as f:
        f.write("{broken")
    error = _set_theme("dracula")
    assert error.startswith("not saved")
    assert theme.active() == "default"


def test_save_optional_writes_through_a_symlink_and_keeps_the_mode(tmp_path):
    real = tmp_path / "dotfiles" / "config.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"base_url": "u", "model": "m"}))
    os.chmod(real, 0o600)
    os.makedirs(config.config_dir(), exist_ok=True)
    os.symlink(real, config.config_path())
    config.save_optional("theme", "gruvbox")
    assert os.path.islink(config.config_path())
    assert json.loads(real.read_text())["theme"] == "gruvbox"
    assert (os.stat(real).st_mode & 0o777) == 0o600


def test_save_optional_refuses_a_config_that_is_not_an_object():
    os.makedirs(config.config_dir(), exist_ok=True)
    with open(config.config_path(), "w") as f:
        f.write("[1, 2]")
    with pytest.raises(ValueError):
        config.save_optional("theme", "dracula")
