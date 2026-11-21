"""commands: the `:`-commands registered via cai.command.

A Command is a named handler registered through the cai.command decorator onto
the current Environment; Environment.load() imports the extensions so theirs
land there too. The tui routes an unrecognized `:`-name to its env's commands(),
calling the fn with a CommandContext - the argument text after the name, the
tab the command was typed in, and all the open tabs.

What a command sees is laid out the way the TUI is:

    ctx.tabs            every open tab
    ctx.tab             the one the command was typed in
        .client         that tab's agent, over the wire
        .screen         that tab's view

ctx.client and ctx.screen are ctx.tab's, for the command that only ever means
"this session". Tabs / Tab hold the few things only the TUI can do for a tab -
run something on the main thread, close it. A command reaches a tab's agent
over its client, the wire, like everything else here; the agent object itself
is not handed out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cai.environment import Environment

if TYPE_CHECKING:
    from cai.screen import Screen
    from cai.tui import AgentClient


@dataclass
class Command:
    """a registered `:command`.

    takes_arg: the palette pre-fills command mode (':merge ') for it instead
               of running it at once.
    idle:      refused, with a note, while the tab's agent is running."""
    fn: object
    help: str = ""
    origin: str = None
    takes_arg: bool = False
    idle: bool = False


def command(fn=None, *, name=None, help="", takes_arg=False, idle=False):
    """decorator: register a `:command` on the current Environment. use bare
    (@cai.command) or with args (@cai.command(name="compact", help="...")). the
    function is the handler, called with a CommandContext; the name defaults to
    the function name. it lands on the env being load()ed - else the process
    default - and a later registration of the same name wins (the user's init.py
    runs last). takes_arg / idle: see Command."""
    def decorator(target):
        cmd_name = name or target.__name__
        origin = None
        code = getattr(target, "__code__", None)
        if code is not None:
            origin = code.co_filename
        Environment.target().register_command(cmd_name,
                                              Command(fn=target,
                                                      help=help,
                                                      origin=origin,
                                                      takes_arg=takes_arg,
                                                      idle=idle))
        return target
    if fn is not None:
        return decorator(fn)
    return decorator


class Tab:
    """one open TUI tab, as a command sees it. two Tab objects for the same
    tab compare equal."""

    def __init__(self, tab, tabs):
        self._tab = tab
        self._tabs = tabs

    def __eq__(self, other):
        if not isinstance(other, Tab):
            return NotImplemented
        return self._tab is other._tab

    def __hash__(self):
        return id(self._tab)

    @property
    def client(self) -> AgentClient:
        """the tab's agent, over the wire."""
        return self._tab.client

    @property
    def screen(self) -> Screen:
        """the tab's own screen; one in the background keeps what is written
        to it and paints it when the tab comes to the front."""
        return self._tab.screen

    @property
    def number(self):
        """the tab's number on the tab line (1-based), or 0 once it is
        closed."""
        return self._tabs.number(self._tab)

    @property
    def busy(self):
        """whether the tab's agent is running a turn."""
        return self._tab.screen.busy

    def call_soon(self, fn):
        """run fn() on the main thread the next time this tab is in front.
        the main thread owns the terminal, so this is how a background thread
        (one waiting on a model call) gets to open the editor or close a tab.
        a tab closed first never runs fn."""
        self._tab.call_soon(fn)


class Tabs:
    """the open TUI tabs, as a command sees them. iterating gives each Tab in
    tab line order."""

    def __init__(self, tabs):
        self._tabs = tabs

    def __len__(self):
        return self._tabs.count()

    def __iter__(self):
        tabs = []
        for tab in self._tabs.all():
            tabs.append(Tab(tab, self._tabs))
        return iter(tabs)

    @property
    def current(self) -> Tab:
        """the tab in front - the one the command was typed in."""
        return Tab(self._tabs.current, self._tabs)

    def at(self, number):
        """the tab with this number on the tab line (1-based), or None."""
        tab = self._tabs.at(number - 1)
        if tab is None:
            return None
        return Tab(tab, self._tabs)

    def close(self, tab):
        """close a tab and its agent. main thread only (a command's fn, or a
        Tab.call_soon). closing a tab in the background leaves the one in
        front where it is; closing the one in front lands on its neighbour.
        the last tab is not closed this way (ValueError) - that is :q."""
        if not tab.number:
            return
        if self._tabs.count() < 2:
            raise ValueError("the last tab cannot be closed")
        self._tabs.close(tab._tab)


@dataclass
class CommandContext:
    """what a command's fn receives: the text after the command name, the tab
    it was typed in, and all the open tabs (tabs.current is tab).

    client and screen are that tab's: the agent client to drive (get_messages
    / set_messages / get_info / submit / steer) and the screen to write back
    to (its get_input / set_input reach the unsent draft in the input box,
    which survives the `:` command). the typed fields let an extension
    author's editor jump from ctx.client.<method> into its definition."""
    args: str
    tab: Tab
    tabs: Tabs

    @property
    def client(self) -> AgentClient:
        return self.tab.client

    @property
    def screen(self) -> Screen:
        return self.tab.screen

    def write(self, text):
        from cai.screen import Screen
        self.screen.write(text, kind=Screen.META)
