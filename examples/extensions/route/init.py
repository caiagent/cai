"""init.py - example extension: pick the model per call, from a hook.

before_turn fires ahead of every model call. the hook reads the run so far
(ctx.usage is the last call's token counts, None before the first) and switches
the agent's model with cai.current_agent().set_model(...). the loop reads the
model right before each call, never from a copy, so the switch is the model
this very call uses.

the rule here: a small model until the prompt grows past LARGE_AT tokens, then
a large one. both ids come from config.json:

    "route_small_model": "...",
    "route_large_model": "..."

with either missing the hook does nothing. :route small|large is the same
switch by hand - a command drives the agent over the wire (ctx.client), the
hook holds the agent itself; both end in Agent.set_model."""
import cai
from cai import config


LARGE_AT = 60_000


@cai.hook("before_turn")
def route(ctx: cai.HookContext):
    agent = cai.current_agent()
    if agent is None: return None
    small = config.load_optional("route_small_model")
    large = config.load_optional("route_large_model")
    if not small or not large: return None
    usage = ctx.usage or {}
    tokens = usage.get("prompt_tokens", 0)
    want = small
    if tokens >= LARGE_AT: want = large
    if agent.model == want: return None
    agent.set_model(want)
    ctx.ui.status(f"route: {want} ({tokens} prompt tokens)")
    return None


@cai.command(name="route", help="route small|large - switch the model by hand")
def route_cmd(ctx: cai.CommandContext):
    which = ctx.args.strip()
    model = config.load_optional(f"route_{which}_model")
    if not model:
        ctx.write("usage: :route small|large (set route_small_model / route_large_model in config.json)\n")
        return
    ctx.client.set_model(model)
    ctx.write(f"model: {model}\n")
