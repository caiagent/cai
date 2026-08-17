"""The child side of cai's `python` tool - the script the tool feeds to the
managed venv's interpreter (as `python -c` SOURCE TEXT, read by cai.pytool;
this module is never imported, and importing it by accident runs nothing).

It reads the snippet from stdin, enters the kernel namespace jail, installs
the audit-hook jail, wires tool_call() over the two inherited RPC fds, and execs
the snippet. stdlib-only, so it runs under the empty managed venv.

The kernel jail (default) pivots into a mount namespace where ONLY cwd +
CAI_SCRATCH + the interpreter prefixes + the system library dirs the dynamic
loader needs (LOADER_DIRS + ld.so.cache) exist - mounted READ-ONLY, except
the write roots, which are re-bound read-write on top - inside an empty
network namespace. The hook jail enforces the same policy in userspace:
reads (and directory listings) are confined to the same roots; writes are
allowed under the write roots only and denied everywhere else, as are
subprocess/exec/fork, sockets, ctypes and cffi. raw os.read/os.write on the
inherited fds emit no audit events and namespaces do not sever inherited
fds, so tool_call() needs no exception in either jail.

CAI_PY_MODE picks the write/exec policy: "read-only" (default) keeps the
scratch dir the one writable island; "read-write" widens the write roots to
the working directory and the CAI_ALLOWED_PATHS grants; "read-write-exec"
additionally allows running programs - the jail then also carries the system
binary dirs and the standard /dev nodes, and a spawned process inherits the
namespaces, so it sees the same files, the same write roots and no network,
kernel-enforced (audit hooks do not survive into an exec'd binary)."""
import sys
import os
import json


# the jail roots - what the snippet may see (kernel jail) and read (audit
# hook), and the subset it may write (per CAI_PY_MODE). computed by main()
# before either jail goes up.
read_roots = []
write_roots = []
mode = "read-only"


def compute_read_roots():
    roots = [os.path.realpath(os.getcwd())]
    scratch = os.environ.get("CAI_SCRATCH", "")
    if scratch:
        roots.append(os.path.realpath(scratch))
    for entry in os.environ.get("CAI_ALLOWED_PATHS", "").split(os.pathsep):
        if not entry: continue
        real = os.path.realpath(entry)
        if real in roots: continue
        roots.append(real)
    for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        real = os.path.realpath(prefix)
        if real in roots: continue
        roots.append(real)
    return roots


def compute_write_roots():
    roots = []
    scratch = os.environ.get("CAI_SCRATCH", "")
    if scratch:
        roots.append(os.path.realpath(scratch))
    if mode == "read-only":
        return roots
    cwd = os.path.realpath(os.getcwd())
    if cwd not in roots:
        roots.append(cwd)
    for entry in os.environ.get("CAI_ALLOWED_PATHS", "").split(os.pathsep):
        if not entry: continue
        real = os.path.realpath(entry)
        if real in roots: continue
        roots.append(real)
    # /dev/null is load-bearing for spawned programs (subprocess.DEVNULL,
    # shell redirections) - a write root so its bind ends up read-write.
    if mode == "read-write-exec":
        roots.append("/dev/null")
    return roots


# --- kernel jail: user+mount+net namespaces --------------------------------
# the mount namespace is pivoted onto a tmpfs holding recursive bind mounts of
# read_roots ONLY, the whole tree then flipped read-only, and the scratch dir
# alone re-bound read-write on top - every other path does not exist and no
# path outside scratch is writable at the kernel level, however the syscall is
# issued. the fresh network namespace has no interfaces (not even loopback) so
# there is no network, and abstract unix sockets are per-netns so they die
# with it. inherited fds (stdin, the RPC pipes, the stdout tempfile) are
# untouched - read-only mounts do not affect already-open fds. runs before
# the audit hook exists, so its own opens/mkdirs are unrestricted.

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
MS_REC = 16384
MS_PRIVATE = 1 << 18
MS_BIND = 4096
MNT_DETACH = 2
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
EINVAL = 22
AT_FDCWD = -100
AT_RECURSIVE = 0x8000
MOUNT_ATTR_RDONLY = 1
MOUNT_SETATTR_NR = 442
PIVOT_ROOT_NR = {}
PIVOT_ROOT_NR["x86_64"] = 155
PIVOT_ROOT_NR["aarch64"] = 41

# the dynamic loader's world. stdlib C extensions (zlib, _ssl, ...) dlopen
# system shared libraries (libz, libssl, ...), and on hosts whose base
# interpreter is not /usr-based (pyenv, uv) those live outside every
# read_root - the import then dies with "libz.so.1: cannot open shared object
# file". bound at their LITERAL paths, because that is how the loader spells
# its search dirs; a merged-usr symlink (/lib -> usr/lib) is recreated as a
# symlink so both spellings resolve. /etc/ld.so.cache rides along so
# distro-specific dirs (multiarch) resolve without the fallback guesswork.
LOADER_DIRS = ("/lib", "/lib64", "/usr/lib", "/usr/lib64")
LOADER_CACHE = "/etc/ld.so.cache"

# the exec world, carried only in read-write-exec mode: the system binary
# dirs (so /bin/sh, git & co exist to be run) and /etc (tools read passwd,
# certs, ...), all read-only - plus the standard /dev nodes programs assume
# (/dev/null joins the write roots instead: it must be writable). /usr first
# so the loader dirs under it fold away.
EXEC_DIRS = ("/usr", "/bin", "/sbin", "/etc")
DEV_NODES = ("/dev/zero", "/dev/urandom", "/dev/random")


def covered(path, roots):
    """True when path is one of roots or sits under one - it then already
    exists in the jail through that root's recursive bind."""
    for root in roots:
        if path == root:
            return True
        if path.startswith(root + os.sep):
            return True
    return False


def bind_roots():
    """read_roots with nested paths folded away - a recursive bind of a parent
    already carries its children."""
    ordered = sorted(read_roots, key=len)
    kept = []
    for root in ordered:
        if covered(root, kept): continue
        kept.append(root)
    return kept


def system_world():
    """the host dirs and files the jail must carry beyond the read roots: the
    dynamic loader's world always; the exec world too when programs may run."""
    dirs = LOADER_DIRS
    files = (LOADER_CACHE,)
    if mode == "read-write-exec":
        dirs = EXEC_DIRS + LOADER_DIRS
        files = ("/dev/null", LOADER_CACHE) + DEV_NODES
    return dirs, files


def bind_system_world(libc, need, staging, kept):
    """make dlopen (and, in exec mode, exec) work inside the jail: bind the
    system_world() dirs and files, skipping whatever a read root's recursive
    bind - or an earlier system dir - already carries. runs before the
    read-only flip, so these end up read-only like everything else (the
    /dev/null write root gets its read-write bind stacked on top later).
    dirs are bound at their LITERAL paths, because that is how the loader and
    PATH spell them; a merged-usr symlink (/bin -> usr/bin) is recreated as a
    symlink so both spellings resolve. files bind onto empty file
    mountpoints."""
    dirs, files = system_world()
    bound = list(kept)
    for sys_dir in dirs:
        if not os.path.lexists(sys_dir): continue
        if covered(sys_dir, bound): continue
        target = staging + sys_dir
        if os.path.islink(sys_dir):
            os.symlink(os.readlink(sys_dir), target)
            bound.append(sys_dir)
            continue
        os.makedirs(target, exist_ok=True)
        need(libc.mount(sys_dir.encode(),
                        target.encode(),
                        None,
                        MS_BIND | MS_REC,
                        None),
             f"bind {sys_dir}")
        bound.append(sys_dir)

    for sys_file in files:
        if not os.path.exists(sys_file): continue
        if covered(sys_file, bound): continue
        target = staging + sys_file
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w"):
            pass
        need(libc.mount(sys_file.encode(), target.encode(), None, MS_BIND, None),
             f"bind {sys_file}")


def drop_caps(ctypes, libc, need):
    """the namespaces above were built with the full capabilities a fresh user
    namespace grants - the snippet must not keep them, or it could rearrange its
    own jail (remount, umount its binds)."""
    need(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)
    cap = 0
    while cap < 64:
        rc = libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
        if rc != 0 and ctypes.get_errno() == EINVAL:
            break
        cap += 1

    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32),
                    ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]

    header = CapHeader()
    header.version = 0x20080522
    header.pid = 0
    data = (CapData * 2)()
    need(libc.capset(ctypes.byref(header), data), "capset")


def enter_kernel_jail():
    import ctypes
    staging = os.environ.get("CAI_PY_STAGING", "")
    if not staging:
        raise RuntimeError("no staging directory was provided")
    machine = os.uname().machine
    if machine not in PIVOT_ROOT_NR:
        raise RuntimeError(f"unsupported architecture {machine!r}")
    libc = ctypes.CDLL(None, use_errno=True)

    def need(rc, what):
        if rc == 0:
            return
        err = ctypes.get_errno()
        raise OSError(err, what + ": " + os.strerror(err))

    uid = os.getuid()
    gid = os.getgid()
    cwd = os.path.realpath(os.getcwd())

    need(libc.unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET), "unshare")

    with open("/proc/self/setgroups", "w") as f:
        f.write("deny")
    with open("/proc/self/uid_map", "w") as f:
        f.write(f"{uid} {uid} 1")
    with open("/proc/self/gid_map", "w") as f:
        f.write(f"{gid} {gid} 1")

    need(libc.mount(b"none", b"/", None, MS_REC | MS_PRIVATE, None), "make / private")
    need(libc.mount(b"tmpfs", staging.encode(), b"tmpfs", 0, None), "mount tmpfs")

    kept = bind_roots()
    for root in kept:
        target = staging + root
        # a file root (an allowed-paths grant) binds onto an empty file
        # mountpoint, like LOADER_CACHE below.
        if not os.path.isdir(root):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w"):
                pass
            need(libc.mount(root.encode(), target.encode(), None, MS_BIND, None),
                 f"bind {root}")
            continue
        os.makedirs(target, exist_ok=True)

        need(libc.mount(root.encode(),
                        target.encode(),
                        None,
                        MS_BIND | MS_REC,
                        None),
             f"bind {root}")
    bind_system_world(libc, need, staging, kept)

    class MountAttr(ctypes.Structure):
        _fields_ = [("attr_set", ctypes.c_uint64),
                    ("attr_clr", ctypes.c_uint64),
                    ("propagation", ctypes.c_uint64),
                    ("userns_fd", ctypes.c_uint64)]

    # MS_RDONLY on the initial bind is silently ignored by the kernel; making
    # a bind read-only takes a second step. one recursive mount_setattr over
    # the staging tmpfs flips it and every bind under it atomically (needs
    # kernel >= 5.12, same era as unprivileged userns hosts that get here).
    attr = MountAttr()
    attr.attr_set = MOUNT_ATTR_RDONLY
    need(libc.syscall(MOUNT_SETATTR_NR,
                      AT_FDCWD,
                      staging.encode(),
                      AT_RECURSIVE,
                      ctypes.byref(attr),
                      ctypes.sizeof(attr)),
        "make jail read-only")
    # the write roots are the writable islands: re-bind each over the
    # read-only tree - a fresh bind mount is read-write by default and stacks
    # on top of the read-only one below. every staging path already exists:
    # a write root is either a read root (bound above or carried inside a
    # parent's recursive bind) or /dev/null (bound by bind_system_world).
    for root in write_roots:
        target = staging + root
        need(libc.mount(root.encode(), target.encode(), None, MS_BIND | MS_REC, None),
             f"bind {root} read-write")
    os.chdir(staging)
    need(libc.syscall(PIVOT_ROOT_NR[machine], b".", b"."), "pivot_root")
    need(libc.umount2(b".", MNT_DETACH), "detach old root")
    os.chdir(cwd)
    drop_caps(ctypes, libc, need)
    # this bootstrap imported ctypes; evict it so the snippet's own import still
    # hits the audit hook instead of the sys.modules cache.
    for name in list(sys.modules):
        if name == "_ctypes" or name == "ctypes" or name.startswith("ctypes."):
            del sys.modules[name]


# --- hook jail: the read-only audit hook ------------------------------------

# process creation and signalling - blocked except in read-write-exec mode,
# where running programs is the point. ctypes/cffi (the FFI_MODULES import
# block) lift with it: once arbitrary binaries may run, blocking in-process
# FFI is theater, and the kernel jail is the boundary either way.
PROCESS_BLOCKED = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "pty.spawn",
)

# blocked in every mode: no mode grants network (the empty netns enforces it
# in the kernel jail; here it is the friendly error).
NETWORK_BLOCKED = (
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.sendmsg",
)

FFI_MODULES = ("ctypes", "_ctypes", "cffi", "_cffi_backend")

WRITE_EVENTS = (
    "os.remove",
    "os.rename",
    "os.mkdir",
    "os.rmdir",
    "os.link",
    "os.symlink",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.utime",
    "shutil.rmtree",
    "shutil.move",
)

# listing a directory's entries is still a read - confine it to read_roots the
# same way open-for-read is, or the jail leaks the whole filesystem tree
# (os.listdir/scandir, and glob/os.walk/pathlib.iterdir which build on scandir).
# note the stat family (os.stat, os.path.exists/getsize) emits NO audit event -
# under the kernel jail that no longer matters (the paths don't exist), but the
# hook-only fallback keeps the historic existence/size residual.
READ_EVENTS = (
    "os.listdir",
    "os.scandir",
)

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

in_hook = [False]


def under(path, roots):
    """True when path resolves into one of roots. non-path args (fds, None)
    pass - the open that produced the fd was already checked."""
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, str):
        return True
    resolved = os.path.realpath(path)
    for root in roots:
        if resolved == root:
            return True
        if resolved.startswith(root + os.sep):
            return True
    return False


def wants_write(mode, flags):
    if isinstance(mode, str):
        for ch in "wax+":
            if ch in mode:
                return True
        return False
    if not isinstance(flags, int):
        return False
    return bool(flags & WRITE_FLAGS)


def write_denial(what):
    if mode == "read-only":
        return f"cai sandbox: writes are allowed under the scratch dir only - {what}"
    return f"cai sandbox: write outside the writable roots (cwd, scratch, granted paths) - {what}"


def check(event, args):
    if event == "import":
        if args[0] not in FFI_MODULES:
            return
        if mode == "read-write-exec":
            return
        raise PermissionError(f"cai sandbox: {args[0]} is blocked")
    if event in NETWORK_BLOCKED:
        raise PermissionError(f"cai sandbox: {event} is blocked")
    if event in PROCESS_BLOCKED:
        if mode == "read-write-exec":
            return
        raise PermissionError(f"cai sandbox: {event} is blocked")
    if event == "open":
        path, open_mode, flags = args
        if wants_write(open_mode, flags):
            if under(path, write_roots):
                return
            raise PermissionError(write_denial(f"cannot open {path!r} for writing"))
        if not under(path, read_roots):
            raise PermissionError(f"cai sandbox: open outside the working directory: {path!r}")
        return
    if event in READ_EVENTS:
        for arg in args:
            if under(arg, read_roots): continue
            raise PermissionError(f"cai sandbox: {event} outside the working directory: {arg!r}")
        return
    if event in WRITE_EVENTS:
        for arg in args:
            if under(arg, write_roots): continue
            raise PermissionError(write_denial(f"{event}: {arg!r}"))


def hook(event, args):
    if in_hook[0]:
        return
    in_hook[0] = True
    try:
        check(event, args)
    finally:
        in_hook[0] = False


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
            raise RuntimeError("cai sandbox: tool channel closed")
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
    global read_roots, write_roots, mode, _RPC_RD, _RPC_WR

    code = sys.stdin.read()
    mode = os.environ.get("CAI_PY_MODE", "read-only")
    read_roots = compute_read_roots()
    write_roots = compute_write_roots()

    if os.environ.get("CAI_PY_SANDBOX", "kernel") != "hook":
        try:
            enter_kernel_jail()
        except Exception as e:
            sys.stderr.write(
                f"cai sandbox: could not enter the kernel namespace jail: {e}\n"
                "this host may not allow unprivileged user namespaces (common under\n"
                "default-hardened containers). if the environment itself is already a\n"
                'boundary, set "python_sandbox": "hook" in ~/.config/cai/config.json\n'
                "to run with the audit-hook sandbox only.\n")
            sys.exit(97)

    _RPC_RD = int(os.environ["CAI_PY_RPC_READ"])
    _RPC_WR = int(os.environ["CAI_PY_RPC_WRITE"])

    snippet_globals = {}
    snippet_globals["__name__"] = "__main__"
    snippet_globals["tool_call"] = tool_call
    snippet_globals.update(tool_globals())

    sys.addaudithook(hook)
    exec(compile(code, "<python>", "exec"), snippet_globals)


if __name__ == "__main__":
    main()
