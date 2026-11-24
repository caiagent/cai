"""tui.commands: the `:`-commands. one table (_Command entries) is the single
definition of a command - its name, palette help, whether it takes an argument
and whether it must wait for an idle agent - and both the palette and the
dispatcher are built from it. the commands of one session are defined here;
the tab commands add theirs in cai.tui.tabs, and cai.tui.app joins the two."""
import logging
import os
from dataclasses import dataclass

from cai.commands import CommandContext
from cai.commands import Tabs
from cai.screen import Screen
from cai.session import SessionsRegistry
from cai.ui import reset_ui
from cai.ui import set_ui
from cai.tui.transcript import _replay_messages
from cai.tui.client import _CONTINUE
from cai.tui.pickers import (
    _load_session,
    _open_agents,
    _open_config,
    _open_history,
    _open_messages,
    _open_requests,
    _open_models,
    _open_pending,
    _open_prompts,
    _open_sessions,
    _open_skills,
    _open_system,
    _open_tools,
)


log = logging.getLogger("cai")


@dataclass(frozen=True)
class _Command:
    """one `:`-command. `run` is called with a _Context and returns True to
    leave the TUI (anything else carries on).

    takes_arg: the palette pre-fills command mode (':save ') for it instead
               of running it at once.
    idle:      refused while a run is in flight - a command that changes the
               conversation, or the tool registry, under the worker's feet.
    aliases:   other names that run it; not listed in the palette.
    listed:    shown in the palette and offered by ':' completion."""
    name: str
    help: str
    run: object
    takes_arg: bool = False
    idle: bool = False
    aliases: tuple = ()
    listed: bool = True


@dataclass
class _Context:
    """what a command's handler gets: the name it was called by, the text
    after it, the tab in front, all the tabs, and what the tabs share."""
    head: str
    arg: str
    tab: object
    tabs: object

    @property
    def screen(self):
        return self.tab.screen

    @property
    def client(self):
        return self.tab.client

    @property
    def status(self):
        return self.tab.status

    @property
    def jobs(self):
        return self.tab.jobs

    @property
    def env(self):
        return self.tab.shared.env

    @property
    def settings(self):
        return self.tab.shared.env.settings

    def note(self, text):
        """a one-line answer to the command, in the conversation view."""
        self.screen.write(f"[{text}]\n", kind=Screen.META, block=True)

    def refuse_busy(self):
        """True, with a note saying so, while a run is in flight."""
        if not self.screen.busy:
            return False
        self.note(f"busy — :{self.head} when idle")
        return True


def _paths_command(screen, client, head, arg):
    """:allow / :disallow. bare shows that side's live list; a comma list of
    files or directories (the flag's syntax) replaces it; a lone '-' clears
    it. a missing entry is reported and changes nothing."""
    side = "allowed"
    if head == "disallow":
        side = "disallowed"
    if arg:
        spec = arg
        if arg == "-":
            spec = ""
        if head == "allow":
            error = client.set_paths(allowed=spec)
        else:
            error = client.set_paths(disallowed=spec)
        if error:
            screen.write(f"[:{head} {error}]\n", kind=Screen.META, block=True)
            return
    roots = client.get_paths().get(side) or []
    if not roots:
        screen.write(f"[{side}: (none)]\n", kind=Screen.META, block=True)
        return
    screen.write(f"[{side}:]\n" + "\n".join(roots) + "\n", kind=Screen.META, block=True)


def _pwd_command(screen, client):
    """:pwd - print the agent's working directory (read over the wire: the
    agent's cwd, which --cwd may have pointed away from the TUI's own)."""
    try:
        cwd = client.get_info().get("cwd", "")
    except OSError as e:
        screen.write(f"[:pwd {e}]\n", kind=Screen.ERROR, block=True)
        return
    screen.write(f"[{cwd or '(unknown)'}]\n", kind=Screen.META, block=True)


def _save_session(screen, client, path, cfg):
    """save the agent (over the wire) to path, or its default <name>.flow when
    path is empty. the written path comes back from the control op."""
    written = client.save(path or None)
    if written is None:
        screen.write("[save error]\n", kind=Screen.ERROR, block=True)
        return
    SessionsRegistry.prune(cfg.max_sessions_mb)
    screen.write(f"[saved {written}]\n", kind=Screen.META, block=True)


def _refresh_tokens(status):
    # resync the ctx readout after a client-side message change (e.g. :clear) the
    # worker is not in the loop for: the official count drops to '?' since no api
    # sample covers the new messages until the next turn produces one.
    status.set_tokens(0)


def _clear(ctx):
    ctx.client.set_messages([])
    ctx.screen.clear_buffer()
    _refresh_tokens(ctx.status)


def _redraw(ctx):
    ctx.screen.clear_buffer()
    _replay_messages(ctx.screen, ctx.client.get_messages(), ctx.settings)


def _paths(ctx):
    # set_paths is a deferred op (servers restart between turns), so a busy
    # agent would hold the reply until the turn ends; showing is always fine.
    if ctx.arg and ctx.refuse_busy():
        return
    _paths_command(ctx.screen, ctx.client, ctx.head, ctx.arg)


def _save(ctx):
    path = ""
    if ctx.arg:
        path = os.path.expanduser(ctx.arg)
    _save_session(ctx.screen, ctx.client, path, ctx.settings)


def _load(ctx):
    if not ctx.arg:
        ctx.note("usage: :load <path>")
        return
    _load_session(ctx.screen, ctx.client, ctx.status, os.path.expanduser(ctx.arg), ctx.settings)


def _continue(ctx):
    if not ctx.client.get_messages():
        ctx.note("nothing to continue")
        return
    ctx.jobs.put(_CONTINUE)


# the commands of one session, in palette order.
_SESSION_COMMANDS = [
    _Command("models", "switch the model (pin favorites)",
             lambda ctx: _open_models(ctx.screen, ctx.client, ctx.status, ctx.tab.shared.registry)),
    _Command("tools", "enable / disable tools",
             lambda ctx: _open_tools(ctx.screen, ctx.client, ctx.settings), idle=True),
    _Command("skills", "activate / deactivate skills",
             lambda ctx: _open_skills(ctx.screen, ctx.client, ctx.settings), idle=True),
    _Command("messages", "view / edit / delete the conversation",
             lambda ctx: _open_messages(ctx.screen, ctx.client, ctx.status)),
    _Command("requests", "the HTTP requests sent to the model server (troubleshooting)",
             lambda ctx: _open_requests(ctx.screen, ctx.tab.shared.requests,
                                        ctx.client.get_info().get("name", ""))),
    _Command("history", "fork back to an earlier point in this conversation",
             lambda ctx: _open_history(ctx.screen, ctx.client, ctx.status, ctx.jobs, ctx.settings),
             idle=True),
    _Command("prompts", "fuzzy-search your prompt history",
             lambda ctx: _open_prompts(ctx.screen)),
    _Command("continue", "continue the conversation without a new prompt", _continue, idle=True),
    _Command("agents", "live view of sub-agents",
             lambda ctx: _open_agents(ctx.screen, ctx.client, ctx.settings,
                                      ctx.tab.shared.requests, ctx.status)),
    _Command("status", "toggle the status panel (Tab): model, cwd, skills, tools, sub-agents",
             lambda ctx: ctx.tab.status_panel.toggle()),
    _Command("pending", "remove queued prompts / steers before they run",
             lambda ctx: _open_pending(ctx.screen, ctx.client, ctx.jobs, ctx.tab.pending)),
    _Command("system", "view / edit the system prompt",
             lambda ctx: _open_system(ctx.screen, ctx.client), idle=True),
    _Command("sessions", "load a saved session",
             lambda ctx: _open_sessions(ctx.screen, ctx.client, ctx.status, ctx.settings), idle=True),
    _Command("config", "edit the live session settings",
             lambda ctx: _open_config(ctx.screen, ctx.client, ctx.settings)),
    _Command("pwd", "print the agent's current working directory",
             lambda ctx: _pwd_command(ctx.screen, ctx.client)),
    _Command("allow", "grant tools extra paths: comma list replaces, bare shows, '-' clears",
             _paths, takes_arg=True),
    _Command("disallow", "deny tools paths everywhere: comma list replaces, bare shows, '-' clears",
             _paths, takes_arg=True),
    _Command("save", "save the session (optional path)", _save, takes_arg=True),
    _Command("load", "load a session file (path)", _load, takes_arg=True, idle=True),
    _Command("clear", "clear the conversation view", _clear, idle=True),
    _Command("redraw", "repaint the view from the conversation", _redraw, idle=True),
]


def _command_index(commands):
    """{name or alias: _Command} for a command table."""
    index = {}
    for command in commands:
        index[command.name] = command
        for alias in command.aliases:
            index[alias] = command
    return index


def _palette_entries(commands):
    """(name, help) for the palette and ':' completion, in table order."""
    entries = []
    for command in commands:
        if not command.listed: continue
        entries.append((command.name, command.help))
    return entries


def _arg_command_names(commands):
    """the commands the palette pre-fills (':save ') instead of running."""
    names = []
    for command in commands:
        if command.takes_arg:
            names.append(command.name)
    return tuple(names)


def _run_extension_command(ctx):
    """a command registered through cai.command (env.commands()). True when
    there is one by this name - whether it ran cleanly, raised, or was
    refused for a running agent (Command.idle)."""
    command = ctx.env.commands().get(ctx.head)
    if command is None:
        return False
    if command.idle and ctx.refuse_busy():
        return True
    token = set_ui(ctx.tab.worker.ui)
    try:
        tabs = Tabs(ctx.tabs)
        command.fn(CommandContext(ctx.arg, tabs.current, tabs))
    except KeyboardInterrupt:
        # Ctrl-C while the command had the main thread (the terminal sends
        # the signal between two prompts): the command is over, the TUI is
        # not.
        ctx.note(f":{ctx.head} interrupted")
    except Exception as e:
        log.exception("command :%s raised", ctx.head)
        ctx.screen.write(f"[command :{ctx.head} failed: {type(e).__name__}: {e}]\n",
                         kind=Screen.ERROR, block=True)
    finally:
        reset_ui(token)
    return True


def _dispatch(index, tabs, cmd):
    """run a `:`-command on the tab in front. cmd is the text after ':' (or
    the name a key binding raised: 'tabnext' for gt); its first word names
    the command, the rest is its argument. a name the table does not have
    falls back to a cai.command extension. returns True to leave the TUI."""
    head, _sep, arg = cmd.partition(" ")
    ctx = _Context(head, arg.strip(), tabs.current, tabs)
    command = index.get(head)
    if command is None:
        if not _run_extension_command(ctx):
            ctx.note(f"unknown command: :{head}")
        return False
    if command.idle and ctx.refuse_busy():
        return False
    return command.run(ctx) is True
