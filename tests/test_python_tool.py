"""Tests for the builtin `python` tool: the audit-hook sandbox, the managed
venv, and the tool_call() tool-proxy that dispatches the agent's own tools in-process
through the run's gates. The child is really spawned; for speed most tests point
the venv at the current interpreter (the sandbox blocks cffi regardless), and one
test exercises real venv creation."""
import os
import sys
import time

import pytest

import cai
from cai import config, hooks, pytool
from cai.agent import Agent
from cai.hooks import HooksRegistry, RunGate


def _fast_venv(monkeypatch):
    """skip real venv creation - run the child under this interpreter."""
    monkeypatch.setattr(pytool, "ensure_venv", lambda: sys.executable)


def _force_sandbox(monkeypatch, mode):
    """pin the python_sandbox mode - keeps the test independent of the
    developer's real config.json."""
    def fake_optional(key, default=None):
        if key == "python_sandbox":
            return mode
        return default
    monkeypatch.setattr(config, "load_optional", fake_optional)


def _run(agent, code, timeout=20):
    return agent.tools_registry.dispatch("python", {"code": code, "timeout": timeout})


def _agent_with_echo():
    @cai.tool
    def echo(x: int) -> str:
        """echo x back."""
        return f"echo:{x}"
    return Agent(model="m", api=object(), tools=["echo"])


# --- wiring ---------------------------------------------------------------

def test_python_registered_unselected_until_the_skill():
    agent = Agent(model="m", api=object())
    try:
        assert "python" in agent.tools_registry.names()
        assert "python" not in agent.tools
    finally:
        agent.close()


def test_python_skills_select_the_tool():
    agent = Agent(model="m", api=object(), skills=["python-read-only"])
    try:
        assert "python" in agent.tools
    finally:
        agent.close()
    # the read-write skill layers on the read-only one, like fs on fs-read-only
    agent = Agent(model="m", api=object(), skills=["python"])
    try:
        assert "python" in agent.tools
        assert "python-read-only" in agent.skills
    finally:
        agent.close()


# --- venv -----------------------------------------------------------------

def test_ensure_venv_creates_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path))
    python = pytool.ensure_venv()
    assert os.path.exists(python)
    assert pytool.ensure_venv() == python


def test_ensure_venv_rebuilds_a_broken_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path))
    python = pytool.ensure_venv()
    os.remove(python)
    assert pytool.ensure_venv() == python
    assert os.path.exists(python)


def test_ensure_venv_uses_a_user_supplied_python_venv(tmp_path, monkeypatch):
    # python_venv points at an existing env: ensure_venv just returns its
    # interpreter, never running `python -m venv`.
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path / "cfg"))
    env = tmp_path / "myenv"
    monkeypatch.setattr(config, "load_optional",
                        lambda key, default=None: str(env) if key == "python_venv" else default)
    monkeypatch.setattr(config, "venv_dir", lambda: str(env))
    built = pytool.subprocess.run([sys.executable, "-m", "venv", str(env)])
    assert built.returncode == 0

    real_run = pytool.subprocess.run
    calls = []

    def record(cmd, *a, **k):
        calls.append(cmd)
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(pytool.subprocess, "run", record)

    assert pytool.ensure_venv() == config.venv_python()
    for cmd in calls:
        assert "venv" not in cmd, f"ensure_venv rebuilt a user-supplied env: {cmd}"


def test_ensure_venv_refuses_to_rebuild_a_broken_python_venv(tmp_path, monkeypatch):
    # a user-supplied env that is missing/unusable raises, never gets wiped.
    monkeypatch.setattr(config, "config_dir", lambda: str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "venv_dir", lambda: str(tmp_path / "gone"))
    monkeypatch.setattr(config, "load_optional",
                        lambda key, default=None: str(tmp_path / "gone") if key == "python_venv" else default)
    with pytest.raises(RuntimeError, match="python_venv"):
        pytool.ensure_venv()


# --- sandbox --------------------------------------------------------------

def test_output_and_exit_code(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print(2 ** 10)").strip() == "1024"
        assert _run(agent, "x = 1") == "(no output)"
        assert "[exit code 3]" in _run(agent, "import sys; sys.exit(3)")
    finally:
        agent.close()


def test_reads_allowed_writes_confined_to_scratch(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (cwd / "existing.txt").write_text("data")
    monkeypatch.chdir(cwd)

    @cai.tool
    def noop() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop"], scratch=str(scratch))
    try:
        # reading inside the jail is fine
        assert _run(agent, "print(open('existing.txt').read())").strip() == "data"
        # writes outside scratch are denied - cwd included
        assert "scratch dir only" in _run(agent, "open('new.txt','w').write('x')")
        assert not (cwd / "new.txt").exists()
        # scratch is the one writable island - the write really lands
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'s.txt');"
                         " open(p,'w').write('y')")
        _run(agent, scratch_write)
        assert (scratch / "s.txt").read_text() == "y"
        # deleting / renaming under scratch works too
        scratch_remove = ("import os; os.remove(os.path.join(os.environ['CAI_SCRATCH'],'s.txt'))")
        _run(agent, scratch_remove)
        assert not (scratch / "s.txt").exists()
        # but no delete / rename / mkdir of anything outside it
        assert "scratch dir only" in _run(agent, "import os; os.remove('existing.txt')")
        assert (cwd / "existing.txt").exists()
        assert "scratch dir only" in _run(agent, "import os; os.rename('existing.txt','r.txt')")
        assert "scratch dir only" in _run(agent, "import shutil; shutil.rmtree('.')")
        # a rename may not smuggle a file across the boundary either way
        cross = ("import os; os.rename('existing.txt',"
                 " os.path.join(os.environ['CAI_SCRATCH'],'stolen.txt'))")
        assert "scratch dir only" in _run(agent, cross)
    finally:
        agent.close()


@pytest.mark.parametrize("snippet", [
    "open('/etc/hostname','w')",
    "open('/etc/hostname')",
    "import socket; socket.socket().connect(('127.0.0.1', 9))",
    # directory enumeration outside the cwd is a read too - must not leak the tree
    "import os; os.listdir('/')",
    "import os; os.scandir('/etc')",
    "import pathlib; list(pathlib.Path('/').iterdir())",
])
def test_sandbox_denials(snippet, tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert "PermissionError" in _run(agent, snippet)
    finally:
        agent.close()


def test_masked_traversal_leaks_nothing(tmp_path, monkeypatch):
    # os.walk / glob swallow the scandir denial (onerror ignores it), so they
    # don't raise - but they must come back EMPTY, never the outside listing.
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "import os; print(list(os.walk('/etc')))").strip() == "[]"
        assert _run(agent, "import glob; print(glob.glob('/etc/*'))").strip() == "[]"
    finally:
        agent.close()


def test_reading_and_listing_allowed_inside_the_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    agent = _agent_with_echo()
    try:
        out = _run(agent, "import os; print(sorted(os.listdir('.')))")
        assert "a.txt" in out
        out = _run(agent, "import glob; print(glob.glob('*.txt'))")
        assert "a.txt" in out
    finally:
        agent.close()


def test_disallowed_paths_are_masked_inside_the_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "key.pem").write_text("private bytes")
    denied_file = tmp_path / "denied.txt"
    denied_file.write_text("also private")
    (tmp_path / "open.txt").write_text("public bytes")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", os.pathsep.join([str(secret), str(denied_file)]))
    # a spawned program sees only the kernel masks (no audit hook), so it
    # proves the masks hold on their own
    agent = _agent_with_skill("python")
    # wrapped in markers: the tool renders empty stdout as "(no output)"
    sh = ("import subprocess; r = subprocess.run(['sh','-c',{cmd!r}],capture_output=True,text=True);"
          " print('<' + r.stdout.strip() + '>')")
    try:
        # the rest of cwd is as usable as ever - even writable in this mode
        assert _run(agent, "print(open('open.txt').read())").strip() == "public bytes"
        assert _run(agent, "open('new.txt','w').write('x'); print('ok')").strip() == "ok"
        # in-process, the audit hook refuses a denied path outright
        out = _run(agent, "print(open('secret/key.pem').read())")
        assert "private bytes" not in out
        assert "disallowed" in out
        # a denied directory is an empty read-only tmpfs: listing shows nothing,
        # its contents cannot be read, nothing can be created inside
        assert _run(agent, sh.format(cmd="ls secret")).strip() == "<>"
        assert _run(agent, sh.format(cmd="cat secret/key.pem")).strip() == "<>"
        assert _run(agent, sh.format(cmd="echo x > secret/new.txt && echo ok")).strip() == "<>"
        assert not (secret / "new.txt").exists()
        # a denied file reads back empty and cannot be written
        assert _run(agent, sh.format(cmd="cat denied.txt")).strip() == "<>"
        assert _run(agent, sh.format(cmd="echo x > denied.txt && echo ok")).strip() == "<>"
        assert denied_file.read_text() == "also private"
        assert (secret / "key.pem").read_text() == "private bytes"
    finally:
        agent.close()


def test_disallowed_paths_refused_by_the_hook_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "hook")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "key.pem").write_text("private bytes")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAI_DISALLOWED_PATHS", str(secret))
    agent = _agent_with_echo()
    try:
        out = _run(agent, "print(open('secret/key.pem').read())")
        assert "private bytes" not in out
        assert "disallowed" in out
        assert "disallowed" in _run(agent, "import os; print(os.listdir('secret'))")
    finally:
        agent.close()


def test_allowed_file_grant_binds_into_the_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    granted = shared / "granted.txt"
    granted.write_text("file grant")
    (shared / "sibling.txt").write_text("not granted")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(granted))
    agent = _agent_with_echo()
    try:
        # the granted file is bound read-only onto a file mountpoint
        assert _run(agent, f"print(open({str(granted)!r}).read())").strip() == "file grant"
        assert "scratch dir only" in _run(agent, f"open({str(granted)!r},'a').write('x')")
        # its siblings do not exist inside the jail
        out = _run(agent, f"print(open({str(shared / 'sibling.txt')!r}).read())")
        assert "not granted" not in out
    finally:
        agent.close()


def test_timeout(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "while True: pass", timeout=1) == "Error: run timed out after 1s"
    finally:
        agent.close()


# --- kernel jail ------------------------------------------------------------

STAT_PROBE = """import os
try:
    os.stat('/root')
    print('visible')
except FileNotFoundError:
    print('hidden')
"""


def test_kernel_jail_outside_paths_do_not_exist(tmp_path, monkeypatch):
    # the stat family emits no audit event - only the kernel jail hides this.
    # the system world (/usr, /etc, the loader dirs) is bound read-only for
    # programs to run - everything else on the host must not exist.
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, STAT_PROBE).strip() == "hidden"
        probe = "import os; print(os.path.exists('/home'), os.path.exists('/var'))"
        assert _run(agent, probe).strip() == "False False"
    finally:
        agent.close()


NET_PROBE = """import socket
s = socket.socket()
try:
    s.connect(('example.com', 80))
    print('connected')
except socket.gaierror:
    print('no-network')
"""


def test_kernel_jail_has_no_network(tmp_path, monkeypatch):
    # hostname resolution runs in libc, below the audit hook - inside the empty
    # network namespace it always fails, however the snippet reaches for the net
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        assert _run(agent, NET_PROBE).strip() == "no-network"
    finally:
        agent.close()


def test_kernel_jail_carries_the_loader_world(tmp_path, monkeypatch):
    # stdlib C extensions dlopen system shared libraries (libssl, libz, ...)
    # that live outside the interpreter prefixes on pyenv/uv-style hosts - the
    # jail must carry the loader's dirs + ld.so.cache. the cache probe is the
    # regression signal: it was absent from the jail before the loader-world
    # binds existed (os.path.isfile is stat-family, no audit event fires).
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    probe = """import os
import _ssl
import zlib
print('extensions-ok', os.path.isfile('/etc/ld.so.cache'))
"""
    try:
        assert _run(agent, probe).strip() == "extensions-ok True"
    finally:
        agent.close()


def test_kernel_jail_still_reads_cwd_and_scratch(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (cwd / "in.txt").write_text("cwd-data")
    (scratch / "s.txt").write_text("scratch-data")
    monkeypatch.chdir(cwd)

    @cai.tool
    def noop2() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop2"], scratch=str(scratch))
    try:
        assert _run(agent, "print(open('in.txt').read())").strip() == "cwd-data"
        scratch_read = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'s.txt');"
                        " print(open(p).read())")
        assert _run(agent, scratch_read).strip() == "scratch-data"
    finally:
        agent.close()


def test_kernel_jail_scratch_is_writable_but_cwd_is_not(tmp_path, monkeypatch):
    # the write must land on the real disk through the read-write scratch bind,
    # while a cwd write still fails (the hook answers first with PermissionError,
    # an OSError; had it been evaded, the read-only mount answers with EROFS -
    # another OSError - so the probe holds for both layers).
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(cwd)

    @cai.tool
    def noop3() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop3"], scratch=str(scratch))
    try:
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'w.txt');"
                         " open(p,'w').write('landed'); print('ok')")
        assert _run(agent, scratch_write).strip() == "ok"
        assert (scratch / "w.txt").read_text() == "landed"
        cwd_probe = """import os
try:
    os.open('raw.txt', os.O_WRONLY | os.O_CREAT)
    print('writable')
except OSError:
    print('read-only')
"""
        assert _run(agent, cwd_probe).strip() == "read-only"
        assert not (cwd / "raw.txt").exists()
    finally:
        agent.close()


def test_kernel_jail_scratch_nested_under_cwd_still_writable(tmp_path, monkeypatch):
    # scratch inside cwd: the recursive cwd bind already carries it, so it gets
    # no read bind of its own - the read-write bind must still stack on top.
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(tmp_path)

    @cai.tool
    def noop4() -> str:
        """noop."""
        return ""
    agent = Agent(model="m", api=object(), tools=["noop4"], scratch=str(scratch))
    try:
        scratch_write = ("import os; p=os.path.join(os.environ['CAI_SCRATCH'],'n.txt');"
                         " open(p,'w').write('nested'); print('ok')")
        assert _run(agent, scratch_write).strip() == "ok"
        assert (scratch / "n.txt").read_text() == "nested"
    finally:
        agent.close()


def test_kernel_jail_staging_dir_is_cleaned_up(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    made = []
    real_mkdtemp = pytool.tempfile.mkdtemp

    def spy_mkdtemp(prefix=None):
        path = real_mkdtemp(prefix=prefix)
        if prefix == "cai-py-":
            made.append(path)
        return path
    monkeypatch.setattr(pytool.tempfile, "mkdtemp", spy_mkdtemp)
    agent = _agent_with_echo()
    try:
        assert _run(agent, "print('ok')").strip() == "ok"
        assert len(made) == 1
        assert not os.path.exists(made[0])
    finally:
        agent.close()


def test_hook_mode_skips_the_kernel_jail(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "hook")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        # the host filesystem is visible again (the documented stat residual)...
        assert _run(agent, STAT_PROBE).strip() == "visible"
        # ...but the audit hook still confines opens
        assert "PermissionError" in _run(agent, "open('/etc/hostname')")
    finally:
        agent.close()


# --- modes ------------------------------------------------------------------

def _agent_with_skill(skill, scratch=None):
    @cai.tool
    def echo(x: int) -> str:
        """echo x back."""
        return f"echo:{x}"
    return Agent(model="m", api=object(), tools=["echo"], skills=[skill],
                 scratch=scratch)


def test_sandbox_mode_follows_the_active_skill():
    agent = Agent(model="m", api=object(), skills=["python-read-only"])
    try:
        assert pytool.sandbox_mode(agent) == "read-only"
    finally:
        agent.close()
    # `python` layers on `python-read-only`; the wider active mode wins
    agent = _agent_with_skill("python")
    try:
        assert "python-read-only" in agent.skills
        assert pytool.sandbox_mode(agent) == "read-write"
    finally:
        agent.close()


def test_read_write_mode_writes_cwd_and_grants(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("CAI_ALLOWED_PATHS", str(shared))
    agent = _agent_with_skill("python")
    try:
        # cwd is writable now - the write lands on the real disk
        assert _run(agent, "open('new.txt','w').write('x'); print('ok')").strip() == "ok"
        assert (cwd / "new.txt").read_text() == "x"
        # an allowed-paths grant is writable too, matching safe_path's policy
        grant_write = f"open({str(shared / 'g.txt')!r},'w').write('y'); print('ok')"
        assert _run(agent, grant_write).strip() == "ok"
        assert (shared / "g.txt").read_text() == "y"
        # outside the write roots stays denied, with the mode's own message
        assert "writable roots" in _run(agent, "open('/etc/hostname','w')")
    finally:
        agent.close()


def test_read_write_runs_programs(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_skill("python")
    probe = """import subprocess
r = subprocess.run(['/bin/sh', '-c', 'echo hello-from-sh'], capture_output=True, text=True)
print(r.stdout.strip())
"""
    write_probe = """import subprocess
subprocess.run(['/bin/sh', '-c', 'echo landed > out.txt'])
print(open('out.txt').read().strip())
"""
    erofs_probe = """import subprocess
r = subprocess.run(['/bin/sh', '-c', 'echo x > /etc/cai-probe'], capture_output=True)
print('denied' if r.returncode != 0 else 'wrote')
"""
    devnull_probe = """import subprocess
r = subprocess.run(['/bin/sh', '-c', 'echo hi > /dev/null && echo devnull-ok'],
                   capture_output=True, text=True)
print(r.stdout.strip())
"""
    try:
        assert _run(agent, probe).strip() == "hello-from-sh"
        # read-write covers spawned programs too: the shell writes the cwd
        assert _run(agent, write_probe).strip() == "landed"
        assert (tmp_path / "out.txt").read_text().strip() == "landed"
        # ...but outside the write roots the kernel answers EROFS, hooks or not
        assert _run(agent, erofs_probe).strip() == "denied"
        # /dev/null exists and is writable - subprocess.DEVNULL & co depend on it
        assert _run(agent, devnull_probe).strip() == "devnull-ok"
    finally:
        agent.close()


def test_hook_layer_allows_subprocess_and_ctypes_in_read_only(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "hook")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    probe = """import subprocess
r = subprocess.run(['/bin/echo', 'ok'], capture_output=True, text=True)
print(r.stdout.strip())
"""
    try:
        assert _run(agent, probe).strip() == "ok"
        # in-process FFI blocking would be theater when binaries may run
        assert _run(agent, "import ctypes; print('ctypes-ok')").strip() == "ctypes-ok"
    finally:
        agent.close()


def _alive(pid):
    """True while pid is a live (non-zombie) process."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return False
    state = data.rsplit(")", 1)[1].split()[0]
    return state != "Z"


def test_daemon_is_swept_after_the_run(tmp_path, monkeypatch):
    # the child leads its own session; the whole process group is killed when
    # the run ends, so a spawned daemon cannot outlive it.
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    probe = """import subprocess
p = subprocess.Popen(['sleep', '60'])
print(p.pid)
"""
    try:
        pid = int(_run(agent, probe).strip())
    finally:
        agent.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _alive(pid):
            break
        time.sleep(0.05)
    assert not _alive(pid)


# --- tool proxy -----------------------------------------------------------

def test_call_proxies_a_tool_and_only_print_reaches_output(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # the raw result is computed but NOT printed; only the derived value is
        out = _run(agent, "r = tool_call('echo', x=7)\nprint(r.split(':')[1])")
        assert out.strip() == "7"
        assert "echo:7" not in out
    finally:
        agent.close()


def test_signatures_lists_selected_tools_but_not_python():
    agent = _agent_with_echo()
    try:
        agent.tools_registry.select("python")
        sigs = agent.tools_registry.signatures()
        assert "echo(x)" in sigs
        assert "python" not in sigs   # hidden: it cannot call itself
    finally:
        agent.close()


def test_skill_prompt_fills_the_tools_slot():
    from cai.skills import SkillsRegistry
    from cai.tools import ToolsRegistry

    @cai.tool
    def echo(x: int) -> str:
        """echo x back."""
        return f"echo:{x}"
    registry = ToolsRegistry.for_tools(["echo"])
    skills = SkillsRegistry(registry)
    skills._skills["s"] = type("S", (), {"body": "tools:\n{{tools}}"})()
    assert "echo(x)" in skills.system_prompt
    assert "{{tools}}" not in skills.system_prompt


def test_selected_tools_are_callable_by_name(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # the tool is in the snippet's namespace as a plain function - no
        # tool_call() needed; only the derived value is printed
        out = _run(agent, "print(echo(x=5).split(':')[1])")
        assert out.strip() == "5"
    finally:
        agent.close()


def test_python_is_not_offered_as_a_callable(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # python must not be a proxy in the namespace (it cannot call itself)
        assert "NameError" in _run(agent, "print(python)")
    finally:
        agent.close()


def test_call_recursion_and_unavailable_guards(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        assert "cannot call itself" in _run(agent, "print(tool_call('python', code='x'))")
        assert "not available" in _run(agent, "print(tool_call('secret'))")
    finally:
        agent.close()


def test_inner_call_is_gated_by_before_tool_call(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()

    def veto(ctx):
        return False
    registry = HooksRegistry()
    registry.register("before_tool_call", veto)
    gate = RunGate(hooks=registry,
                   dispatch=agent.tools_registry.dispatch,
                   model="m",
                   config=None,
                   ui=hooks.NULL_UI,
                   messages=[],
                   usage=None,
                   hooks_data={})
    token = hooks.set_gate(gate)
    try:
        out = _run(agent, "print(tool_call('echo', x=1))")
        assert "aborted by a before_tool_call hook" in out
    finally:
        hooks.reset_gate(token)
        agent.close()


def test_inner_call_without_a_gate_still_dispatches(monkeypatch):
    _fast_venv(monkeypatch)
    agent = _agent_with_echo()
    try:
        # no run gate published (SDK-style direct use): confined dispatch, ungated
        assert hooks.current_gate() is None
        assert _run(agent, "print(tool_call('echo', x=9))").strip() == "echo:9"
    finally:
        agent.close()


def test_install_runs_pip_in_the_managed_venv(monkeypatch):
    monkeypatch.setattr(pytool, "ensure_venv", lambda: "/venv/bin/python")
    recorded = []

    class FakeResult:
        returncode = 3

    def fake_run(cmd):
        recorded.append(cmd)
        return FakeResult()
    monkeypatch.setattr(pytool.subprocess, "run", fake_run)

    assert pytool.install(["requests", "numpy>=2"]) == 3
    assert recorded == [["/venv/bin/python", "-m", "pip", "install",
                         "requests", "numpy>=2"]]


def test_uninstall_runs_pip_without_prompting(monkeypatch):
    monkeypatch.setattr(pytool, "ensure_venv", lambda: "/venv/bin/python")
    recorded = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd):
        recorded.append(cmd)
        return FakeResult()
    monkeypatch.setattr(pytool.subprocess, "run", fake_run)

    assert pytool.uninstall(["requests"]) == 0
    assert recorded == [["/venv/bin/python", "-m", "pip", "uninstall", "-y",
                         "requests"]]


def test_cli_python_subcommands_dispatch(monkeypatch):
    from cai import cli
    recorded = []

    def fake_install(packages):
        recorded.append(("install", packages))
        return 0

    def fake_uninstall(packages):
        recorded.append(("uninstall", packages))
        return 0

    def fake_list_packages():
        recorded.append(("list-packages", None))
        return 0
    monkeypatch.setattr(pytool, "install", fake_install)
    monkeypatch.setattr(pytool, "uninstall", fake_uninstall)
    monkeypatch.setattr(pytool, "list_packages", fake_list_packages)

    assert cli.main(["python", "install", "requests"]) == 0
    assert cli.main(["python", "uninstall", "requests"]) == 0
    assert cli.main(["python", "list-packages"]) == 0
    assert recorded == [("install", ["requests"]),
                        ("uninstall", ["requests"]),
                        ("list-packages", None)]


def test_kernel_jail_cannot_signal_host_processes(tmp_path, monkeypatch):
    # os.kill is not hook-gated - the pid namespace is what keeps a snippet from
    # signalling cai (or anything else of the user's): host pids do not exist in
    # there, and the snippet is pid 1 of its own namespace
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    probe = f"""import os
try:
    os.kill({os.getpid()}, 0)
    print('reachable')
except ProcessLookupError:
    print('unreachable')
print(os.getpid())
"""
    try:
        assert _run(agent, probe).split() == ["unreachable", "1"]
    finally:
        agent.close()


def test_kernel_jail_relays_exit_code_across_the_pid_namespace(tmp_path, monkeypatch):
    _fast_venv(monkeypatch)
    _force_sandbox(monkeypatch, "kernel")
    monkeypatch.chdir(tmp_path)
    agent = _agent_with_echo()
    try:
        out = _run(agent, "import sys; print('bye'); sys.exit(3)")
        assert "bye" in out
        assert "[exit code 3]" in out
    finally:
        agent.close()
