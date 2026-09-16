name: python
skills: python-read-only
tools: python
---
# Skill: Python (Read-Write)

Adds write access to `python-read-only`: a snippet - and any program it runs -
may now WRITE under the current-working-directory and the granted paths as
well as scratch, the same policy the fs tools enforce. Nothing else is
writable, and nothing else exists.

Discipline:
- Read before you write; confirm a write landed.
- Prefer scratch for intermediates you don't want in the project tree.
- Make the smallest change that does the task.
