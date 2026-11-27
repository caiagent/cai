"""tui.app: run() - build the first tab and drive the main loop: prompt, then
route what came back to a tab command, a `:`-command, a steer or a new turn."""
import atexit
import signal
import sys
import threading

from cai.environment import Environment
from cai.screen import Screen
from cai import sessions_index
from cai import theme
from cai.session import SessionsRegistry
from cai.tui.status import _DEFAULT_CONTEXT_SIZE, _show_agent_errors
from cai.tui.pickers import _load_session, _open_sessions
from cai.tui.commands import (
    _SESSION_COMMANDS,
    _arg_command_names,
    _command_index,
    _dispatch,
    _palette_entries,
)
from cai.tui.requests import RequestStore
from cai.tui.tabs import _TAB_COMMANDS, _Shared, _Tab, _Tabs


# every builtin `:`-command: the session's, then the tabs'. the palette (and
# ':' completion) and the dispatcher are both built from this one table.
_COMMANDS = _SESSION_COMMANDS + _TAB_COMMANDS
_PALETTE_COMMANDS = _palette_entries(_COMMANDS)
_PALETTE_ARG_COMMANDS = _arg_command_names(_COMMANDS)


def _exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def run(*,
        model=None,
        system_prompt=None,
        tools=None,
        skills=None,
        reasoning_effort=None,
        temperature=None,
        max_steps=None,
        resume_path=None,
        pick_session=False,
        initial_prompt=None):
    """launch the interactive TUI around a fresh in-process Agent. blocks until
    the user quits (:q, Ctrl-C, or EOF). returns the process exit code.

    resume_path loads that saved session at startup (--continue); pick_session
    opens the :sessions picker at startup (--sessions). either resumes the chosen
    session in place, so autosave writes back to it.

    initial_prompt, when given, is submitted as the first user turn at startup
    (e.g. `cai -i -- hello`) - the TUI comes up, runs it, then takes over input."""
    from cai import config
    from cai.agent import Agent
    from cai.hooks import HookEvent
    from cai.models import ModelsRegistry

    try:
        env = Environment.default().load()
    except ValueError as e:
        # a cai.sandbox that covers nothing (Environment.check_sandboxes):
        # said before the screen comes up, where it can be read.
        print(f"cai: {e}", file=sys.stderr)
        return 2
    # the cai.settings skills / tools are auto-activated on every CLI run, merged
    # in on top of any --skill / --tool the user passed.
    skills = Environment.merge_activations(skills, env.settings.skills)
    tools = Environment.merge_activations(tools, env.settings.tools)

    # autosave: persist the session to <name>.flow on every conversation
    # mutation, driven solely by hooks - AFTER_RUN (the final answer landed) and
    # MESSAGES_MUTATED (tool results appended mid-run, plus out-of-run edits like
    # a :history fork) together cover every change a run makes (a no-tool turn
    # ends in AFTER_RUN; a tool step appends results and fires MESSAGES_MUTATED).
    # loading a session fires MESSAGES_LOADED, not MESSAGES_MUTATED, so a plain
    # load - whose messages came straight off disk - is not written back and does
    # not bump the file's mtime. the agent comes through ctx.data, and the hook
    # runs on the worker thread mid-run, so it sees a consistent conversation and
    # never races the main thread.
    def _autosave(ctx):
        if not env.settings.auto_save_sessions: return
        saved = (ctx.data or {}).get("agent")

        if saved is None: return
        if not SessionsRegistry.has_real_messages(saved.get_messages()): return

        try:
            saved.save()
            SessionsRegistry.prune(env.settings.max_sessions_mb)
        except OSError:
            pass

    autosave_hooks = []
    autosave_hooks.append((HookEvent.AFTER_RUN, _autosave))
    autosave_hooks.append((HookEvent.MESSAGES_MUTATED, _autosave))

    # the :requests view: the loop keeps no record of its HTTP requests, it
    # only announces them - this hook is what remembers the recent ones. a
    # cloned tab's agent and every sub-agent inherit the hook, so one store
    # holds them all.
    requests = RequestStore(env.settings)
    request_hooks = []
    request_hooks.append((HookEvent.AFTER_REQUEST, requests.on_request))

    agent = Agent(model=model,
                  env=env,
                  system_prompt=system_prompt,
                  tools=tools or [],
                  skills=skills or [],
                  hooks=autosave_hooks + request_hooks,
                  reasoning_effort=reasoning_effort,
                  temperature=temperature,
                  max_steps=max_steps,
                  tool_result_max_chars=env.settings.tool_result_max_chars)

    from cai.api import OpenAiApi
    cfg = config.load_config()
    fallback_limit = int(config.load_optional("default_context_size", _DEFAULT_CONTEXT_SIZE))
    registry = ModelsRegistry(OpenAiApi(cfg.base_url,
                                        config.load_api_key(),
                                        ssl_verify=config.load_optional("ssl_verify", True)))
    palette = list(_PALETTE_COMMANDS)
    arg_commands = list(_PALETTE_ARG_COMMANDS)
    commands = env.commands()
    for name in sorted(commands):
        palette.append((name, commands[name].help))
        if commands[name].takes_arg:
            arg_commands.append(name)
    completions = {}
    for name, _help in palette:
        completions[name] = []
    shared = _Shared(env, registry, fallback_limit, palette, tuple(arg_commands), completions,
                     requests)
    index = _command_index(_COMMANDS)

    tabs = _Tabs()
    try:
        tab = _Tab(agent, shared, tabs)
    except OSError as e:
        # no keyboard to drive the TUI from (-i forced on a run with neither a
        # terminal on stdin nor a /dev/tty): say so instead of tracing back.
        print(f"cai: {e.strerror or e}", file=sys.stderr)
        return 1
    tabs.add(tab)
    screen = tab.screen
    client = tab.client
    status = tab.status

    # warm the catalogue cache in the background so the current model's
    # context_length is saved and the ctx % can derive from it - without
    # blocking startup or the first prompt. models() respects the daily cache,
    # so this is a network call at most once a day.
    def _warm_models():
        registry.models()
        status.refresh_limit()

    threading.Thread(target=_warm_models, daemon=True, name="cai-tui-models-warm").start()

    # read the saved sessions in the background now, so :sessions opens on a
    # full index instead of reading them while the user waits.
    sessions_index.default().warm()

    # the saved colour theme, now the first screen owns the terminal. undone in
    # the finally below, and at exit in case something escapes it.
    atexit.register(theme.reset)
    # a plain `kill` would skip both and leave the terminal recoloured: turn it
    # into an ordinary exit, unless something else already handles it.
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _exit_on_signal)
    try:
        theme.apply(config.load_optional("theme", theme.DEFAULT))
    except ValueError as e:
        screen.write(f"[theme: {e}]\n", kind=Screen.META, block=True)

    screen.write(f"cai — model {tab.model}. i to type; :q to quit; Tab for status.\n",
                 kind=Screen.META, block=True)
    # what failed to load before the TUI attached, then the poll loop takes over.
    _show_agent_errors(screen, client)
    tab.poll_loop.start()

    # resume at startup: --continue loads the resolved session directly;
    # --sessions opens the picker. nothing is running yet, so no idle gate.
    if resume_path:
        _load_session(screen, client, status, resume_path, env.settings)
    elif pick_session:
        _open_sessions(screen, client, status, env.settings)

    # a prompt passed on the command line with -i: hand it to the worker as the
    # first turn. the worker echoes the USER event and streams the answer just
    # as if the user had typed it, then the input loop below takes over.
    if initial_prompt and initial_prompt.strip():
        tab.pending.user_queued()
        tab.jobs.put(initial_prompt)

    try:
        while tabs.count():
            # the tab in front: every pass re-reads it, since a command below
            # may have switched, opened or closed one.
            tab = tabs.current
            # what a background thread left for this tab's main thread (a
            # merge summary ready for review); it may close the tab.
            while tab.todo and tabs.count() and tabs.current is tab:
                tab.todo.popleft()()
            if not tabs.count() or tabs.current is not tab:
                continue
            screen = tab.screen
            user_input = screen.prompt("> ")
            if user_input is None:
                continue
            cmd = screen.pop_command()
            if cmd is not None:
                if _dispatch(index, tabs, cmd):
                    break
                continue
            # '!text' steers the in-flight run; with nothing running it is just
            # an ordinary prompt with the bang stripped.
            if user_input.startswith("!"):
                steer_text = user_input[1:]
                if screen.busy:
                    tab.client.steer(steer_text)
                    continue
                user_input = steer_text
            if not user_input.strip():
                continue
            tab.pending.user_queued()
            tab.jobs.put(user_input)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        theme.reset()
        if tabs.count():
            tabs.close_all()
    return 0
