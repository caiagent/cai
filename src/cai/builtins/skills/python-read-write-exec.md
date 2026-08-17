name: python-read-write-exec
tools: python
---
# Skill: Python (read-write, exec)

`python(code="...", timeout=60)` runs a snippet and returns its stdout/stderr.
Use it when a task is easier as a few lines of code than as a chain of tool
calls. `print()` what you want back.

## Rules

- One-shot: fresh interpreter every call, no variables survive between calls.
- File-system access - READ and WRITE - at the current-working-directory,
  scratch (`os.environ['CAI_SCRATCH']`) and granted paths.
- Running programs is allowed (`subprocess` & co). A spawned program runs
  inside the same sandbox: it sees the same files, writes only where you can
  write, and has no network. `ps`-style tools don't work (no `/proc`);
  a program needing a temp dir gets scratch via `TMPDIR`.
- No network. `input()` sees EOF.
- Never `print()` data and re-submit it to a tool - your context is precious.
  If the data itself is not what you care about but its transfer between A to
  B is, dont print - simply chain provided tools directly in your code.

## Call your dedicated tools from Python

Your tools are already in the snippet's namespace as plain functions —
call them directly:

```
{{tools}}
```

Arguments are text only, NEVER Python `bytes`. Encode binary as text first:

```python
fs__create_file(file_path="C", content=data.hex(), encoding="hex")   # right
fs__create_file(file_path="C", content=data)                         # WRONG: bytes
```

`tool_call(name, **kwargs) -> str` does the same thing by name — use it when the
tool name is dynamic (never `python`, which cannot call itself).
