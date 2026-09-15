# ask

Tools the model calls to question the user mid-run: an open question
(`ask__ask`), pick one (`ask__choose`), pick many (`ask__choose_many`).

This is the smallest real example of **`cai.current_ui()`**: the UI of the run
that dispatched the tool. The same accessor works from a hook or a `:` command,
so any in-process extension reaches the human through one object. In the tui
the questions open the confirm/select/toggle/text overlays; on the CLI they
prompt on stdin; a served agent forwards them to the attached client. With no
human reachable (`ui.interactive` is False) the tools return an `Error:` line
and the model carries on.

Install and use:

    cai extend examples/extensions/ask
    cai --skill ask -- set up the project the way I like it

Layout:

    ask/
    ├── README.md
    ├── init.py          # @cai.tool ask / choose / choose_many
    └── skills/ask.md    # the skill that hands the tools to the model
