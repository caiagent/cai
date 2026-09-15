name: ask
tools: ask__ask, ask__choose, ask__choose_many
---
# Skill: Ask

You can question the user mid-task instead of guessing.

Tools:
- `ask__ask` — an open question, answered in free text.
- `ask__choose` — the user picks exactly one option.
- `ask__choose_many` — the user picks any number of options.

Ask when a decision is genuinely the user's to make and the answer changes what
you do next. Keep questions short, offer concrete options, never ask what you
can find out yourself. An `Error:` result means no user could answer — proceed
on the sensible default and say which one you took.

Example:
    ask__choose(question="Which test runner?", options=["pytest", "unittest"])
