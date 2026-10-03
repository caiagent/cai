"""cai - a small LLM agent, built from scratch layer by layer.

Layer 0: cai.api    - the OpenAI-compatible HTTP client (the LLM call).
Layer 1: cai.llm    - the core agentic loop (call_llm).
         cai.decision - the System One decision call (decide): typed
                        answers about a state, no loop.
         cai.events - the Event value the loop yields, and EventType.
         cai.hooks  - the hook registry the loop fires.
Layer 2: cai.agent  - Agent (persistent conversation) + Run (one-shot execution).
Entry:   cai.config      - bootstrap settings (API key, OpenRouter endpoint).
         cai.environment - the loaded install catalogue (tools/hooks/commands/
                           settings) an Agent resolves against.
         cai.cli         - the `cai` command: prompt in, streamed answer out.
"""
import logging
from typing import TYPE_CHECKING

# every module logs through getLogger("cai"); point that logger (never the
# root) at a file so the diagnostics land somewhere readable without touching
# a host app's logging - its records don't leak in, cai's don't leak out.
_log = logging.getLogger("cai")
if not _log.handlers:
    _handler = logging.FileHandler("/tmp/cai.log")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False

from cai.paths import safe_path, scratch_dir
from cai.ui import current_ui
from cai.events import Event, EventType
from cai.hooks import HookContext, HookEvent, HookResult, HooksRegistry, ToolCall, hook, current_agent
from cai.commands import Command, CommandContext, command
from cai.skills import SlotContext, slot
from cai.llm import LLMError, MaxStepsReached, call_llm

# Agent/Run/Environment stay lazy at runtime (see __getattr__ below) so
# `import cai` doesn't pull agent + its config/api. this block is type-checker
# only - it never runs - so an editor resolves cai.Agent / cai.Run to their real
# definitions (go-to-def) without paying the import.
if TYPE_CHECKING:
    from cai.agent import Agent, Run, RunInFlight
    from cai.api import ApiError, SystemOneApi
    from cai.decision import decide
    from cai.environment import Environment, Settings
    from cai.tools import ToolsRegistry, tool, wrap, mcp_server
    # cai.settings is served by __getattr__ at runtime; this declaration is
    # what lets an editor complete cai.settings and its attributes.
    settings: Settings

__all__ = [
    "safe_path",
    "scratch_dir",
    "current_ui",
    "Event",
    "EventType",
    "HookContext",
    "HookEvent",
    "HookResult",
    "HooksRegistry",
    "ToolCall",
    "hook",
    "Command",
    "CommandContext",
    "command",
    "SlotContext",
    "slot",
    "ToolsRegistry",
    "tool",
    "wrap",
    "current_agent",
    "mcp_server",
    "ApiError",
    "SystemOneApi",
    "decide",
    "LLMError",
    "MaxStepsReached",
    "call_llm",
    "Agent",
    "Run",
    "RunInFlight",
    "Environment",
    "Settings",
    "settings",
]


def __getattr__(name):
    # lazy so `import cai` doesn't pull agent (and its config/api) unless used.
    if name == "Agent":
        from cai.agent import Agent
        return Agent
    if name == "Run":
        from cai.agent import Run
        return Run
    if name == "RunInFlight":
        from cai.agent import RunInFlight
        return RunInFlight
    if name == "Environment":
        from cai.environment import Environment
        return Environment
    if name == "Settings":
        from cai.environment import Settings
        return Settings
    # api pulls requests; lazy so `import cai` stays light.
    if name == "ApiError":
        from cai.api import ApiError
        return ApiError
    if name == "SystemOneApi":
        from cai.api import SystemOneApi
        return SystemOneApi
    # decision pulls api (requests) and config; lazy for the same reason.
    if name == "decide":
        from cai.decision import decide
        return decide
    # tools pulls the environment, so keep it lazy too - cai.tool resolves
    # here the first time an extension's tools/*.py decorates a function.
    if name == "tool":
        from cai.tools import tool
        return tool
    if name == "wrap":
        from cai.tools import wrap
        return wrap
    if name == "mcp_server":
        from cai.tools import mcp_server
        return mcp_server
    if name == "ToolsRegistry":
        from cai.tools import ToolsRegistry
        return ToolsRegistry
    # cai.settings: the current Environment's live Settings - the env being
    # load()ed when one is (so an extension/init.py tunes its own env), else the
    # process default's (the one the :config overlay edits).
    if name == "settings":
        from cai.environment import Environment
        return Environment.target().settings
    raise AttributeError(f"module 'cai' has no attribute {name!r}")
