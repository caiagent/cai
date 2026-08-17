# memory

A persistent note store the model reads for free: a `{{memory__notes}}` slot
in the skill body pushes the latest entries into the system prompt at the
start of every turn. The model never calls a tool to *read* its memory — only
to change it (`memory__add_note` / `memory__remove_note`).

This is the smallest real example of a **slot**: a `@cai.slot` function whose
result fills a `{{name}}` hole in a skill body, resolved fresh each turn.
Anything with queryable state — a task list, a build status, a watched file —
can push itself to the model the same way.

Install and use:

    cai extend examples/extensions/memory
    cai --skill memory -- remember that I prefer tabs over spaces

Notes persist in `~/.config/cai/memory-notes.json` across sessions. Note that
a system prompt that changes between turns invalidates prompt-prefix caching
for the conversation — cheap here (notes change rarely), but keep slot content
stable when nothing happened.
