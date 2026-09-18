"""Tests for the :allow / :disallow TUI commands: bare shows the live list,
a comma list replaces it, '-' clears it, a bad entry is reported."""
from cai.tui import _paths_command


class FakeClient:
    def __init__(self):
        self.allowed = []
        self.disallowed = []
        self.calls = []

    def get_paths(self):
        return {"allowed": list(self.allowed), "disallowed": list(self.disallowed)}

    def set_paths(self, allowed=None, disallowed=None):
        self.calls.append((allowed, disallowed))
        if allowed == "/nope":
            return "entry does not exist: '/nope'"
        if allowed is not None:
            self.allowed = []
            if allowed:
                self.allowed = [allowed]
        if disallowed is not None:
            self.disallowed = []
            if disallowed:
                self.disallowed = [disallowed]
        return None


class FakeScreen:
    def __init__(self):
        self.out = []

    def write(self, text, kind=None, block=False):
        self.out.append(text)


def test_bare_shows_none():
    client = FakeClient()
    screen = FakeScreen()
    _paths_command(screen, client, "allow", "")
    assert screen.out == ["[allowed: (none)]\n"]
    assert client.calls == []


def test_spec_replaces_and_shows_the_list():
    client = FakeClient()
    screen = FakeScreen()
    _paths_command(screen, client, "allow", "/data")
    assert client.calls == [("/data", None)]
    assert screen.out == ["[allowed:]\n/data\n"]


def test_dash_clears_that_side_only():
    client = FakeClient()
    client.allowed = ["/data"]
    client.disallowed = ["/secret"]
    screen = FakeScreen()
    _paths_command(screen, client, "disallow", "-")
    assert client.calls == [(None, "")]
    assert client.allowed == ["/data"]
    assert screen.out == ["[disallowed: (none)]\n"]


def test_bad_entry_is_reported_and_changes_nothing():
    client = FakeClient()
    screen = FakeScreen()
    _paths_command(screen, client, "allow", "/nope")
    assert screen.out == ["[:allow entry does not exist: '/nope']\n"]
    assert client.allowed == []
