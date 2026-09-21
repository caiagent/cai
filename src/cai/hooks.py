"""hooks: the per-run registry plus the shapes passed to every hook.

A hook is an (event, fn) pair. HooksRegistry.fire(event, ctx) calls every fn
registered for that event, synchronously, in registration order, and returns
ONE merged HookResult. A hook returns None to observe, or a HookResult to act:

    return cai.HookResult.veto("would destroy user data")   # before_tool_call
    return cai.HookResult.answer(ctx.content + "\n-- cai")  # on_final_response

Each event supports some of its fields (SUPPORTED_RESULT_FIELDS, and the table
on HookResult); a field an event does not support, and any return that is not a
HookResult, is logged and ignored - the same treatment as an exception in a hook, which is logged and
skipped so one bad hook never breaks the turn.

HookContext is the one shape every hook receives; ToolCall is the slice it
carries for the *_tool_call events. ctx.ui is the live frontend during a run
(so a hook can prompt the human) and a no-op UI otherwise. Fields that don't
apply to the firing event are None. while a hook runs, cai.current_agent() is
the Agent the run belongs to - the same call a tool makes - so a hook acts on
the agent (set_model, set_messages) without knowing how it arrived."""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from cai.environment import Environment
from cai.ui import UI, NULL_UI, reset_ui, set_ui


log = logging.getLogger("cai")


class HookEvent(str, Enum):
    """Every hook event the loop fires, in one place. A str-Enum, so a user
    hook can register with the plain string ('before_tool_call') while core
    code uses the member (HookEvent.BEFORE_TOOL_CALL) - they compare equal."""
    BEFORE_TURN = "before_turn"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    MESSAGES_MUTATED = "messages_mutated"
    MESSAGES_LOADED = "messages_loaded"
    AFTER_TURN = "after_turn"
    ON_FINAL_RESPONSE = "on_final_response"
    AFTER_RUN = "after_run"

    def __str__(self):
        return self.value


VALID_HOOK_EVENTS = tuple(e.value for e in HookEvent)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str  # raw JSON string, as the model emitted it
    args: dict      # parsed (best-effort; {} on parse failure)
    id: str


@dataclass(frozen=True)
class HookResult:
    """what a hook returns to act - one flat type for every event, built with a
    constructor named for the action:

        HookResult.veto(reason="")   before_tool_call: skip the tool; the model
                                     sees `reason` in place of the tool result
        HookResult.answer(content)   on_final_response: replace the answer
        HookResult.finish(content)   after_turn: end the run now with `content`
                                     as its answer (no further model call; the
                                     normal final path still runs -
                                     on_final_response, transcript, after_run)

    the fields each event supports:

        before_tool_call    vetoed, reason
        on_final_response   content
        after_turn          stop, content
        every other event   nothing (observers)

    an observer acts by mutating instead - before_turn fires ahead of every
    model call, so cai.current_agent().set_model(...) there is the model that
    call uses.

    a set field the event does not support is logged and ignored. across several
    hooks on one event: vetoed and stop are or'd; reason and content take the
    last one set. None (return nothing) is the observer's result."""
    vetoed: bool = False
    reason: str = ""
    content: Optional[str] = None
    stop: bool = False

    @classmethod
    def veto(cls, reason=""):
        return cls(vetoed=True, reason=reason)

    @classmethod
    def answer(cls, content):
        return cls(content=content)

    @classmethod
    def finish(cls, content):
        return cls(stop=True, content=content)


# the HookResult fields each event supports - the same table as the HookResult
# docstring, in code. an event not listed supports none (an observer); a set
# field outside its event's entry is ignored with a warning. reason rides with
# vetoed, and on after_turn content rides with stop - alone they mean nothing.
SUPPORTED_RESULT_FIELDS = {}
SUPPORTED_RESULT_FIELDS[HookEvent.BEFORE_TOOL_CALL] = ("vetoed", "reason")
SUPPORTED_RESULT_FIELDS[HookEvent.ON_FINAL_RESPONSE] = ("content",)
SUPPORTED_RESULT_FIELDS[HookEvent.AFTER_TURN] = ("stop", "content")


def _fields_in_use(result):
    """the fields a HookResult carries a value in - the ones fire() checks
    against the event's SUPPORTED_RESULT_FIELDS."""
    out = []
    if result.vetoed: out.append("vetoed")
    if result.reason: out.append("reason")
    if result.content is not None: out.append("content")
    if result.stop: out.append("stop")
    return out


@dataclass(frozen=True)
class HookContext:
    """Passed to every hook, one shape across all events. a hook returns None
    to observe, or a HookResult (HookResult.veto / .answer) to act."""
    event: str
    messages: list
    model: str
    config: Optional[Mapping] = None
    ui: UI = NULL_UI
    usage: Optional[dict] = None
    tool_call: Optional[ToolCall] = None
    content: Optional[str] = None
    data: Optional[dict] = None   # one unified bag: the caller's hooks_data
                                  # (identical across every context of a run)
                                  # with this event's own keys layered on top.


class HooksRegistry:
    """a plain collection of (event, fn) pairs, fired in registration order.
    the install-wide hooks live on the Environment (cai.hook registers there);
    an Agent composes env hooks + its own hooks= list into one of these per
    run."""

    def __init__(self):
        self._entries = []

    @classmethod
    def from_list(cls, hooks):
        """build a registry from a list of (event, fn) pairs (None -> empty)."""
        registry = cls()
        if hooks is None:
            return registry
        for event, fn in hooks:
            registry.register(event, fn)
        return registry

    def register(self, event, fn):
        if event not in VALID_HOOK_EVENTS:
            raise ValueError(f"unknown hook event: {event!r}. valid: {VALID_HOOK_EVENTS}")
        self._entries.append((event, fn))

    def unregister(self, event, fn):
        for i, entry in enumerate(self._entries):
            if entry[0] != event: continue
            if entry[1] != fn: continue
            del self._entries[i]
            return

    def pairs(self):
        """the (event, fn) pairs a child/clone would inherit."""
        out = []
        for event, fn in self._entries:
            out.append((event, fn))
        return out

    def fire(self, event, ctx):
        """call every hook registered for `event` and return one HookResult
        merged from what they returned (see HookResult for the rules). a
        hook that raises is logged and skipped; a return that is not None or
        a HookResult, or a field this event does not support, is logged and
        ignored."""
        vetoed = False
        reason = ""
        content = None
        stop = False
        supported = SUPPORTED_RESULT_FIELDS.get(event, ())
        token = set_ui(ctx.ui)
        agent_token = None
        agent = None
        if ctx.data is not None: agent = ctx.data.get("agent")
        if agent is not None: agent_token = set_agent(agent)
        try:
            for hook_event, fn in self._entries:
                if hook_event != event: continue
                name = getattr(fn, '__name__', repr(fn))
                try:
                    response = fn(ctx)
                except Exception:
                    log.exception("hook %r for %s raised", name, event)
                    continue
                if response is None: continue
                if not isinstance(response, HookResult):
                    log.warning("hook %r for %s returned %r, not a HookResult - ignored",
                                name, event, response)
                    continue
                for field in _fields_in_use(response):
                    if field in supported: continue
                    log.warning("hook %r for %s set HookResult.%s, which %s does not "
                                "support - ignored", name, event, field, event)
                if "vetoed" in supported and response.vetoed:
                    vetoed = True
                    if response.reason: reason = response.reason
                if "stop" in supported and response.stop:
                    stop = True
                    if response.content is not None: content = response.content
                if "stop" in supported and response.content is not None and not response.stop:
                    log.warning("hook %r for %s set HookResult.content without stop - "
                                "ignored (use HookResult.finish)", name, event)
                if "stop" not in supported and "content" in supported and response.content is not None:
                    content = response.content
        finally:
            if agent_token is not None: reset_agent(agent_token)
            reset_ui(token)
        return HookResult(vetoed=vetoed, reason=reason, content=content, stop=stop)


# the agent the current code runs for: ToolsRegistry publishes it around each
# in-process tool call and HooksRegistry.fire around each hook (like
# paths._scratch_provider and ui.current_ui), so a plain @cai.tool or @cai.hook
# reaches the live Agent by importing cai. a ContextVar keeps two agents on two
# threads isolated.
_current_agent = ContextVar("cai_current_agent", default=None)


def current_agent():
    """the Agent whose tool or hook is running right now, or None outside both
    (a bare registry, a command - which drives the agent over the wire through
    ctx.client instead - or an MCP server: another process, which gets the
    agent's name as CAI_AGENT). read it inside a @cai.tool or @cai.hook:

        @cai.hook("before_turn")
        def route(ctx):
            cai.current_agent().set_model("cheap-model")

    note: a thread a tool or hook spawns itself starts with a fresh context -
    capture the agent before spawning."""
    return _current_agent.get()


def set_agent(agent):
    """publish `agent` as the current one; returns the token reset_agent takes."""
    return _current_agent.set(agent)


def reset_agent(token):
    _current_agent.reset(token)


# the run-scoped state an in-process tool needs to dispatch another tool the way
# the loop does. a ContextVar - not a global - so two agents dispatching on two
# threads stay isolated, matching paths._scratch_provider.
_run_gate = ContextVar("cai_run_gate", default=None)


@dataclass(frozen=True)
class RunGate:
    """what call_llm publishes around its dispatch loop so an in-process tool can
    call another tool on the agent's behalf and still be gated. a tool that
    dispatches for the model (the python tool's tool_call()) reads it with
    current_gate() and routes through gated_dispatch, so an inner call fires the
    same before/after_tool_call hooks a top-level call does - a gate can veto it -
    without the inner result ever entering the conversation."""
    hooks: HooksRegistry
    dispatch: object   # callable(name, args) -> str, the run's tools_dispatch
    model: object      # a model id, or a getter - resolved via current_model
    config: Optional[Mapping]
    ui: UI
    messages: list
    usage: Optional[dict]
    hooks_data: Optional[dict]


def set_gate(gate):
    """publish the run gate for the current context; returns a reset token."""
    return _run_gate.set(gate)


def reset_gate(token):
    _run_gate.reset(token)


def current_model(model):
    """the model id right now: `model` itself, or what it returns when it is a
    getter (an Agent's live model). every consumer - the api call, each hook
    context, the run gate - resolves through here at its own moment, never
    from a copy, so a switch made anywhere (a hook, a :model command, another
    thread) is seen by the very next read."""
    if callable(model): return model()
    return model


def current_tools(tools):
    """the tool schemas right now: `tools` itself, or what it returns when it is
    a getter (an Agent's live selection); an empty list reads as None so the
    api omits the tools field. resolved before every model call, never copied,
    so set_selected_tools from a hook is seen by the next call - the same
    contract as current_model."""
    if callable(tools): tools = tools()
    if not tools: return None
    return tools


def current_system_prompt(system_prompt):
    """the system prompt right now: `system_prompt` itself, or what it returns
    when it is a getter (an Agent's live base + active skills, slots filled).
    resolved before every model call, never copied, so set_system_prompt_base
    / set_selected_skills from a hook, and a slot filler's latest state, reach
    the next call - the same contract as current_model."""
    if callable(system_prompt): return system_prompt()
    return system_prompt


def veto_message(name, reason=""):
    """the tool result the model sees for a vetoed call - the fixed wording,
    plus the hook's reason when it gave one."""
    text = f"Error: tool '{name}' was aborted by a before_tool_call hook"
    if reason: text = f"{text}: {reason}"
    return text


def current_gate():
    """the run gate for the current context, or None outside a run loop."""
    return _run_gate.get()


def gated_dispatch(gate, name, args, call_id="tool"):
    """dispatch one tool through the before/after_tool_call hooks: fire before (a
    veto returns the same Error string the loop uses),
    dispatch, fire after, return the result. unlike the loop it does NOT append a
    tool message or fire messages_mutated - an inner call's result goes back to
    its caller, never into the conversation."""
    tool_call = ToolCall(name=name, arguments=json.dumps(args), args=args, id=call_id)
    data = dict(gate.hooks_data or {})
    before = HookContext(event=HookEvent.BEFORE_TOOL_CALL,
                         messages=gate.messages,
                         model=current_model(gate.model),
                         config=gate.config,
                         ui=gate.ui,
                         usage=gate.usage,
                         tool_call=tool_call,
                         data=data)
    result = gate.hooks.fire(HookEvent.BEFORE_TOOL_CALL, before)
    if result.vetoed:
        return veto_message(name, result.reason)
    result = gate.dispatch(name, args)
    if result is None:
        result = ""
    result = str(result)
    after = HookContext(event=HookEvent.AFTER_TOOL_CALL,
                        messages=gate.messages,
                        model=current_model(gate.model),
                        config=gate.config,
                        ui=gate.ui,
                        usage=gate.usage,
                        tool_call=tool_call,
                        content=result,
                        data=data)
    gate.hooks.fire(HookEvent.AFTER_TOOL_CALL, after)
    return result


def hook(event):
    """decorator: register a hook for `event` on the current Environment, e.g.
    @cai.hook("after_turn"). it lands on the env being load()ed - else the
    process default - so once Environment.load() imports the extensions every
    run of an agent on that env fires it. see Environment / HooksRegistry."""
    if event not in VALID_HOOK_EVENTS:
        raise ValueError(f"unknown hook event: {event!r}. valid: {VALID_HOOK_EVENTS}")

    def decorator(fn):
        Environment.target().register_hook(event, fn)
        return fn
    return decorator
