"""pytool: the builtin `python` tool - run a Python snippet, with a callback
into the agent's own tools.

`python(code, timeout=60)` runs a snippet in a subprocess of a cai-MANAGED
virtualenv (config.venv_dir(), created lazily, empty by default - stdlib only,
so its package surface is well-defined), or of the user's own env when
`python_venv` names one. The child script (pytool_bootstrap.py, fed to the
interpreter as `python -c` source text) reads the snippet, wires tool_call()
and execs it.

There is ONE python tool and it has no modes. What a snippet may read, write
and reach is decided by the sandbox the child is started behind:

    cai.sandbox("python", ["bwrap", "--ro-bind", "/", "/", "--bind", "{scratch}", "{scratch}"])

puts that command in front of the interpreter's (see cai.tools.sandbox and
sandboxed_command), and the sandbox alone is the boundary - cai neither looks
inside it nor adds a jail of its own. With no sandbox registered the snippet
is an ordinary process of the user's: the whole filesystem, the network,
every other process. The --allowed-paths / --disallowed-paths grants are the
fs tools' policy and do not reach a snippet either way.

The snippet is also handed a `tool_call(name, **kwargs) -> str` builtin: it names one
of the agent's OWN selected tools, and the call is dispatched IN the cai process
(so every tool keeps its own confinement and reaches live agent state), through
the run's before/after_tool_call hooks (so a gate still vetoes it), and the
result string comes back into the snippet. Only what the snippet print()s becomes
the tool result - so a script can read a big tool result, reduce it in Python,
and return just the answer, the intermediate data never entering model context.
The RPC pipes are plain inherited fds, which a sandbox does not sever - so the
tool_call() channel reaches OUT of the sandbox: a sandboxed snippet can still
have cai run any tool the agent has selected.

The tool is bound to its Agent (like the sub-agent tools) - that is what gives
tool_call() a live dispatch. It is registered on every agent but only offered to the
model when the python skill selects it."""

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import cai
from cai import config
from cai import hooks
from cai.tools import sandboxed_command


PY_TOOL_NAME = "python"


# the child bootstrap lives in pytool_bootstrap.py - a real module so it
# reads and edits like code - but it is executed as SOURCE TEXT via
# `python -c`, never imported: -c keeps sys.path[0] = cwd (so a snippet can
# import the project's own modules), and the file's side effects sit behind
# a __main__ guard, which -c satisfies.
def _bootstrap_source():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytool_bootstrap.py")
    with open(path) as f:
        return f.read()


def _truncate(text):
    """keep a long run's head and tail - tracebacks live at the tail."""
    limit = 20000
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return text[:half] + f"\n[... {omitted} chars omitted ...]\n" + text[-half:]


_venv_lock = threading.Lock()


def _venv_usable(python):
    """True when `python` exists and answers a trivial probe - guards against a
    half-built or stale venv."""
    if not os.path.exists(python):
        return False
    try:
        probe = subprocess.run([python, "-c", "import sys"],
                               capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def ensure_venv():
    """the venv's python, materialized on first use. lazy and idempotent -
    created under a lock so two agents first-using at once don't collide.

    a user-supplied `python_venv` is never built, rebuilt or deleted here: it
    is the user's own env (e.g. a pyenv one), so a broken one raises rather
    than getting silently wiped. only the cai-managed ~/.config/cai/venv is
    created."""
    python = config.venv_python()
    if _venv_usable(python):
        return python
    if config.load_optional("python_venv"):
        raise RuntimeError(
            f"configured python_venv interpreter is missing or unusable: {python}\n"
            "cai does not build or repair a user-supplied env - check the path, or "
            "drop the python_venv setting to use the managed venv.")
    with _venv_lock:
        if _venv_usable(python):
            return python
        base = config.load_optional("python_base", sys.executable)
        shutil.rmtree(config.venv_dir(), ignore_errors=True)
        subprocess.run([base, "-m", "venv", config.venv_dir()], check=True)
        return config.venv_python()


def install(packages):
    """`cai python install` - pip-install packages into the managed venv, from
    outside the sandbox (which may well have no network)."""
    python = ensure_venv()
    result = subprocess.run([python, "-m", "pip", "install"] + list(packages))
    return result.returncode


def uninstall(packages):
    """`cai python uninstall` - pip-uninstall packages from the managed venv.
    -y because there is no terminal to confirm on when scripted."""
    python = ensure_venv()
    result = subprocess.run([python, "-m", "pip", "uninstall", "-y"] + list(packages))
    return result.returncode


def list_packages():
    """`cai python list-packages` - the packages installed in the managed venv."""
    python = ensure_venv()
    result = subprocess.run([python, "-m", "pip", "list"])
    return result.returncode


def _dispatch_gated(agent, name, kwargs):
    """dispatch a tool the snippet asked for, on the agent's behalf. refuses the
    python tool itself (recursion) and any tool not currently offered to the
    model, then routes through the run gate so a before_tool_call gate can veto -
    falling back to a plain confined dispatch outside a run (SDK use)."""
    if not name:
        return "Error: tool request had no name"
    if name == PY_TOOL_NAME:
        return f"Error: {PY_TOOL_NAME} cannot call itself"
    if name not in agent.tools:
        return f"Error: tool {name!r} is not available to tool_call() - callable tools are the agent's selected ones"
    gate = hooks.current_gate()
    if gate is None:
        return agent.tools_registry.dispatch_selected(name, kwargs)
    return hooks.gated_dispatch(gate, name, kwargs, call_id=PY_TOOL_NAME)


def _handle_request(agent, line, rep_w):
    """serve one child tool_call() request: dispatch it and write the reply back."""
    try:
        request = json.loads(line.decode("utf-8"))
    except ValueError:
        result = "Error: malformed tool request"
    else:
        result = _dispatch_gated(agent, request.get("name"), request.get("kwargs") or {})
    reply = {}
    reply["result"] = result
    os.write(rep_w, json.dumps(reply).encode("utf-8") + b"\n")


def _kill_tree(proc):
    """kill the child and everything it spawned: the child leads its own
    session (start_new_session), so its process group is the whole tree - an
    spawned daemon or a timed-out grandchild can't outlive the run. a group
    already gone raises, which is the good case."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def _serve(agent, proc, req_r, rep_w, timeout):
    """single-threaded driver: while the child runs, serve its tool_call() requests on
    this (the run-loop) thread - so inner calls are dispatched exactly like
    top-level ones. returns "timeout" if the child was killed on timeout,
    "interrupted" if killed because the run's interrupt Event was set, None
    when it ended on its own. the child's stdout/stderr go to a temp file, not
    a pipe, so this loop never blocks on output and only ever reads the
    request channel."""
    deadline = time.monotonic() + timeout
    interrupt = getattr(agent, "interrupt", None)
    buf = bytearray()
    while True:
        if interrupt is not None and interrupt.is_set():
            _kill_tree(proc)
            return "interrupted"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_tree(proc)
            return "timeout"
        ready, _, _ = select.select([req_r], [], [], min(remaining, 0.2))
        if req_r not in ready:
            if proc.poll() is not None:
                return None
            continue
        chunk = os.read(req_r, 65536)
        if not chunk:
            return None
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[:nl + 1]
            _handle_request(agent, line, rep_w)


def _run_python(agent, code, timeout):
    """spawn the child - behind the python sandbox, when one is registered -
    feed it the snippet, serve its tool calls, and return its combined
    stdout/stderr (truncated), with an exit-code footer."""
    python = ensure_venv()
    scratch = cai.scratch_dir()

    req_r, req_w = os.pipe()   # child -> parent: tool_call() requests
    rep_r, rep_w = os.pipe()   # parent -> child: results
    # what the child cannot run without. kept apart from the inherited
    # environment because a sandbox may not pass that one on - see
    # sandboxed_command.
    own_env = {}
    if scratch:
        own_env["CAI_SCRATCH"] = scratch
        # a sandbox's /tmp may be missing, private or read-only - point
        # tempfile (and spawned programs honoring TMPDIR) at scratch.
        own_env["TMPDIR"] = scratch
    own_env["CAI_PY_RPC_READ"] = str(rep_r)
    own_env["CAI_PY_RPC_WRITE"] = str(req_w)
    # the agent's own selected tools, minus python itself: the child turns each
    # into a directly-callable function in the snippet's namespace (layer that
    # makes the modified env look ordinary - fs__read_file(...) just works),
    # resolved per call so it tracks the live selection.
    names = []
    for name in agent.tools:
        if name == PY_TOOL_NAME: continue
        names.append(name)
    own_env["CAI_PY_TOOLS"] = json.dumps(names)
    child_env = dict(os.environ)
    child_env.update(own_env)

    out = tempfile.TemporaryFile()
    proc = None
    ended = None
    try:
        command = [python, "-c", _bootstrap_source()]
        sandbox = agent.env.sandbox_for(PY_TOOL_NAME)
        if sandbox is not None:
            command = sandboxed_command(sandbox, command, env=own_env, scratch=scratch)
        proc = subprocess.Popen(command,
                                stdin=subprocess.PIPE,
                                stdout=out,
                                stderr=subprocess.STDOUT,
                                pass_fds=(req_w, rep_r),
                                start_new_session=True,
                                env=child_env)
    finally:
        # the child owns its ends now; the parent keeps req_r/rep_w. closed even
        # on a spawn failure (a sandbox command that does not exist) so no fd
        # leaks.
        os.close(req_w)
        os.close(rep_r)
        if proc is None:
            os.close(req_r)
            os.close(rep_w)
            out.close()

    try:
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.close()
        ended = _serve(agent, proc, req_r, rep_w, timeout)
        proc.wait()
    finally:
        _kill_tree(proc)
        os.close(req_r)
        os.close(rep_w)

    if ended == "timeout":
        out.close()
        return f"Error: run timed out after {timeout}s"
    if ended == "interrupted":
        out.close()
        return "Error: run interrupted by user"

    out.seek(0)
    text = out.read().decode("utf-8", errors="replace")
    out.close()
    text = _truncate(text.rstrip("\n"))
    if proc.returncode != 0:
        if text:
            text += "\n"
        text += f"[exit code {proc.returncode}]"
    if not text:
        return "(no output)"
    return text


_DOC = """Run a Python snippet and return its output (stdout + stderr).

    One-shot: fresh interpreter every call, no variables survive between calls.

    YOUR OWN tools are callable right in the snippet: each is a plain function
    of the same name (e.g. fs__read_file(file_path='big.log')), and
    tool_call(name, **kwargs) does the same by name.

    Where the snippet runs - which files it can read and write, whether it has
    network - is described by the python skill. The session scratch dir
    (os.environ['CAI_SCRATCH']) is the place for intermediates.

    Args:
        code:    Python source to execute.
        timeout: Seconds before the run is killed (default 60).
    """


def make_python(agent):
    """build the python tool bound to `agent`; tool_call() reaches the agent's live
    tools/dispatch through the closure."""
    def python(code: str, timeout: int = 60) -> str:
        return _run_python(agent, code, timeout)
    python.__doc__ = _DOC
    # the concrete tool list the model sees comes from the python skill's
    # {{tools}} slot (registry.signatures); python must not list itself there -
    # it cannot call itself (see _dispatch_gated).
    python._cai_hide_from_tool_list = True
    # the one function tool that is a process of its own - what
    # cai.sandbox("python", ...) covers and the status panel groups it under.
    python._cai_tool_process = PY_TOOL_NAME
    return python


def python_tools(agent):
    """the python tool(s) bound to `agent` - the agent-tools factory analog of
    subagent_tools. registered unselected on every agent; the python skill
    selects it."""
    return [make_python(agent)]
