"""init.py - example extension: refuse tool calls with HookResult.veto.

Two before_tool_call hooks, two ways to answer the same event:

- deny_destructive refuses a fixed set of tools outright. the model reads the
  reason in place of the tool result and carries on - it can pick another way.
- protect_paths asks the human (ctx.ui.confirm) before a write tool touches a
  protected path, and vetoes when nobody is reachable to say yes.

Both return None to let a call through - a hook that returns nothing observes."""
import cai


DESTRUCTIVE = ("fs__remove_file", "fs__move_directory")

WRITE_TOOLS = ("fs__create_file",
               "fs__edit_file",
               "fs__rename_file",
               "fs__move_file",
               "fs__copy_file",
               "fs__remove_file",
               "fs__create_directory",
               "fs__move_directory")

PROTECTED = (".git", ".env", "secrets")


@cai.hook("before_tool_call")
def deny_destructive(ctx: cai.HookContext):
    """refuse the destructive tools no matter what."""
    name = ctx.tool_call.name
    if name not in DESTRUCTIVE: return None
    return cai.HookResult.veto(f"{name} is disabled by the veto extension; leave that to the user")


@cai.hook("before_tool_call")
def protect_paths(ctx: cai.HookContext):
    """ask before a write tool touches a protected path; veto when nobody can answer."""
    name = ctx.tool_call.name
    if name not in WRITE_TOOLS: return None
    path = _path_argument(ctx.tool_call.args)
    if path is None: return None
    if not _is_protected(path): return None
    if not ctx.ui.interactive:
        return cai.HookResult.veto(f"{path} is protected and no user is reachable to allow it")
    allowed = ctx.ui.confirm(f"{name} on protected path {path}?", default=False)
    if allowed: return None
    return cai.HookResult.veto(f"the user refused {name} on {path}")


def _path_argument(args):
    """the first path-like argument of a tool call, or None."""
    for key in ("path", "src_path", "dst_path"):
        value = args.get(key)
        if value: return value
    return None


def _is_protected(path):
    for part in path.replace("\\", "/").split("/"):
        if part in PROTECTED: return True
    return False
