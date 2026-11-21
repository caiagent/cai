"""tui: the interactive, full-screen terminal UI.

Wraps the reused screen/ package (an alternate-screen, vim-modal viewport) around
an Agent that is served over a unix socket (UnixWiredAgent) and driven purely as
a wire client through AgentClient - the TUI never touches the agent object once it
is served, so the same client surface will drive a remote agent over --attach. The
main thread owns the terminal and blocks in Screen.prompt(); a worker thread
SUBMITs prompts over the client's streaming connection and renders the EVENT/RESULT
stream into the viewport - draining it at all times, so a turn another client
submits or steers on the same socket renders here too. Control ops (messages,
tools, skills, model, save, load) go over a separate, lock-guarded control
connection.

The status line follows the reference's pattern: the worker only updates raw
signals (busy, the kind of the last delta, when the last token arrived); a
single status thread samples them every _REFRESH_INTERVAL and renders the line,
so a stream that goes quiet falls back to 'waiting' on its own without any event
to trigger it.

The `:`-commands cover the session: :models / :messages / :history / :sessions /
:save / :load, plus :tools and :skills to toggle which tools and skills the agent
uses, and :redraw to repaint the viewport from the conversation (e.g. after
flipping 'show reasoning' in :config). Each picker is a screen overlay; the
mutating ones (:messages, :history, :sessions, :load, :tools, :skills) and
:redraw are refused while a run is in flight so they never race the worker's
view of the conversation or the tool registry.

The package, bottom up - each module imports only from the ones above it:

  transcript  painting a conversation into a Screen (live stream and replay)
  client      AgentClient, ScreenUI, the jobs queue and the worker thread
  status      status line, chips, the Tab status panel, the poll loop
  pickers     the overlays the commands open (:messages, :sessions, ...)
  commands    the `:`-command table and its dispatcher
  tabs        tabs, the tab line, branching a tab (:checkout)
  app         run(): setup and the main loop

Everything is re-exported here, so `from cai.tui import AgentClient` (and the
tests' private names) keep working.
"""


from cai.tui.transcript import (  # noqa: F401
    _AttachView,
    _Transcript,
    _replay_messages,
    _stored_call_args,
)
from cai.tui.client import (  # noqa: F401
    AgentClient,
    ScreenUI,
    _CONTINUE,
    _Jobs,
    _PUMP_TIMEOUT,
    _Worker,
    _agent_control,
)
from cai.tui.status import (  # noqa: F401
    _AGENTS_REFRESH,
    _AgentChips,
    _DEFAULT_CONTEXT_SIZE,
    _NOTE_SECONDS,
    _Pending,
    _PollLoop,
    _REFRESH_INTERVAL,
    _STALL_SECONDS,
    _STATUS_LABEL_WIDTH,
    _STATUS_MAX_WIDTH,
    _STATUS_MIN_WIDTH,
    _Status,
    _StatusLoop,
    _StatusPanel,
    _agent_chip_lines,
    _clip_head,
    _clip_tail,
    _pack_names,
    _pending_chip_lines,
    _running_subagents,
    _show_agent_errors,
    _status_panel_lines,
    _status_rows,
    _status_text,
)
from cai.tui.pickers import (  # noqa: F401
    _PREVIEW_MESSAGES,
    _attach_agent,
    _bool_setting,
    _conversation_tail,
    _history_node_color,
    _history_node_label,
    _history_node_preview,
    _int_setting,
    _list_setting,
    _load_session,
    _open_agents,
    _open_config,
    _open_history,
    _open_messages,
    _open_models,
    _open_pending,
    _open_prompts,
    _open_sessions,
    _open_skills,
    _open_system,
    _open_tools,
    _pending_nodes,
    _prompt_entries,
    _session_tree_nodes,
    _system_preview,
    _system_prompt_nodes,
    _tool_label,
)
from cai.tui.commands import (  # noqa: F401
    _Command,
    _Context,
    _SESSION_COMMANDS,
    _arg_command_names,
    _clear,
    _command_index,
    _continue,
    _dispatch,
    _load,
    _palette_entries,
    _paths,
    _paths_command,
    _pwd_command,
    _redraw,
    _refresh_tokens,
    _run_extension_command,
    _save,
    _save_session,
)
from cai.tui.tabs import (  # noqa: F401
    _Shared,
    _TAB_COMMANDS,
    _TAB_CURRENT,
    _TAB_FILL,
    _TAB_LABEL_WIDTH,
    _TAB_OTHER,
    _Tab,
    _Tabs,
    _clone_point,
    _close_tab,
    _open_clone_tab,
    _tab_label,
    _tab_line_cells,
)
from cai.tui.app import (  # noqa: F401
    _COMMANDS,
    _PALETTE_ARG_COMMANDS,
    _PALETTE_COMMANDS,
    run,
)
