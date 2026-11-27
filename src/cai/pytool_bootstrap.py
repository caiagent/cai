"""The child side of cai's `python` tool - the script the tool feeds to the
venv's interpreter (as `python -c` SOURCE TEXT, read by cai.pytool; this
module is never imported, and importing it by accident runs nothing).

It reads the snippet from stdin, wires tool_call() over the two inherited RPC
fds, and execs the snippet. stdlib-only, so it runs under the empty managed
venv.

Nothing here confines the snippet: that is the job of the sandbox the whole
process was started behind (cai.sandbox("python", ...)), or of nothing when
none is registered. The RPC fds are plain inherited pipes, which a sandbox
built from namespaces does not sever, and their numbers arrive in
CAI_PY_RPC_READ / CAI_PY_RPC_WRITE."""
import sys
import os
import json


# --- the tool_call() tool proxy ----------------------------------------------

_RPC_RD = -1
_RPC_WR = -1
_rpc_buf = bytearray()


def _rpc_readline():
    while True:
        nl = _rpc_buf.find(b"\n")
        if nl >= 0:
            line = bytes(_rpc_buf[:nl])
            del _rpc_buf[:nl + 1]
            return line
        chunk = os.read(_RPC_RD, 65536)
        if not chunk:
            raise RuntimeError("cai: tool channel closed")
        _rpc_buf.extend(chunk)


def tool_call(name, **kwargs):
    """dispatch one of the agent's own tools and return its result string. the
    call runs in the cai process, through cai's tool gates; only what you
    print() reaches the model, so reduce a big result here first."""
    request = {}
    request["name"] = name
    request["kwargs"] = kwargs
    os.write(_RPC_WR, json.dumps(request).encode("utf-8") + b"\n")
    reply = json.loads(_rpc_readline().decode("utf-8"))
    return reply["result"]


def make_tool_proxy(name):
    """a plain function that dispatches tool `name` over the same RPC as
    tool_call - so a snippet can call it directly (fs__read_file(...))."""
    def proxy(**kwargs):
        return tool_call(name, **kwargs)
    proxy.__name__ = name
    proxy.__doc__ = f"call your {name!r} tool in cai: {name}(**kwargs) -> str"
    return proxy


def tool_globals():
    """the agent's selected tools as directly-callable names for the snippet's
    namespace - each a thin proxy over tool_call. this is what makes the modified
    env look ordinary: the functions are simply there. a name that is not a valid
    identifier is skipped and stays reachable via tool_call by name."""
    names = json.loads(os.environ.get("CAI_PY_TOOLS", "[]"))
    proxies = {}
    for name in names:
        if not name.isidentifier(): continue
        proxies[name] = make_tool_proxy(name)
    return proxies


def main():
    global _RPC_RD, _RPC_WR

    code = sys.stdin.read()
    _RPC_RD = int(os.environ["CAI_PY_RPC_READ"])
    _RPC_WR = int(os.environ["CAI_PY_RPC_WRITE"])

    snippet_globals = {}
    snippet_globals["__name__"] = "__main__"
    snippet_globals["tool_call"] = tool_call
    snippet_globals.update(tool_globals())

    exec(compile(code, "<python>", "exec"), snippet_globals)


if __name__ == "__main__":
    main()
