"""init.py - example extension: a memory skill built on a slot.

The point of this example is the {{memory__notes}} slot in skills/memory.md:
the note store is *pushed* into the system prompt at the start of every turn,
so the model always sees the latest notes without ever calling a read tool.
The tools only mutate the store; reading is free.

Notes persist in ~/.config/cai/memory-notes.json across sessions."""
import os
import json

import cai


def _store_path():
    return os.path.expanduser(os.path.join("~", ".config", "cai", "memory-notes.json"))


def _read_notes():
    try:
        with open(_store_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _write_notes(notes):
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(notes, f, indent=2)


@cai.slot
def notes(ctx):
    """fills {{memory__notes}} with the store's entries, numbered."""
    entries = _read_notes()
    if not entries:
        return "(no notes yet)"
    lines = []
    for i, entry in enumerate(entries, 1):
        lines.append(f"{i}. {entry}")
    return "\n".join(lines)


@cai.tool
def add_note(note: str) -> str:
    """Append a note to the persistent memory store."""
    entries = _read_notes()
    entries.append(note)
    _write_notes(entries)
    return f"noted ({len(entries)} notes total)"


@cai.tool
def remove_note(number: int) -> str:
    """Delete a note by its number, as shown in the notes list."""
    entries = _read_notes()
    if number < 1 or number > len(entries):
        return f"Error: no note {number} (the store has {len(entries)})"
    removed = entries.pop(number - 1)
    _write_notes(entries)
    return f"removed: {removed}"
