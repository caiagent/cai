name: memory
tools: memory__add_note, memory__remove_note
---
# Skill: Memory

You have a persistent note store. Its current entries are shown below —
refreshed at the start of every turn, so what you see is always the latest
state. You never need a tool to *read* your notes; they are simply here.

Tools:
- `memory__add_note` — append a note. Keep each note one self-contained fact.
- `memory__remove_note` — delete a note by its number, when it is wrong or obsolete.

Save what will matter beyond this conversation: user preferences, decisions
and their reasons, project constraints. Don't save what the conversation
already carries or what you could re-derive.

## Current notes
{{memory__notes}}
