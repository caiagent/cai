name: python-read-write
tools: python
---
# Skill: Python (read-write)

`python(code="...", timeout=60)` runs a snippet and returns its stdout/stderr.
Use it when a task is easier as a few lines of code than as a chain of tool
calls. `print()` what you want back.

## Rules

- One-shot: fresh interpreter every call, no variables survive between calls.
- File-system access - READ and WRITE - at the current-working-directory,
  scratch (`os.environ['CAI_SCRATCH']`) and granted paths. Nothing else
  exists.
- No network. No subprocesses. `input()` sees EOF.
- Use scratch for intermediate state you don't want in the project tree.
    - To change or perform any action outside of the python environment - use
      the provided dedicated tools.
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
