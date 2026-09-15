"""init.py - example extension: tools that let the model question the user.

Every tool reaches the human through cai.current_ui() - the UI of the run that
dispatched the tool (the tui's overlays, the CLI's stdin, or a served agent's
client over the wire). with no human reachable a question is answered by the
UI's default, so the model always gets a string back and never hangs."""
import cai


@cai.tool
def ask(question: str) -> str:
    """Ask the user an open question and return their free-text answer."""
    ui = cai.current_ui()
    if not ui.interactive:
        return "Error: no user is reachable to answer"
    answer = ui.text(question)
    if answer is None:
        return "Error: the user did not answer"
    return answer


@cai.tool
def choose(question: str, options: list[str]) -> str:
    """Ask the user to pick exactly one of the options; returns the chosen option."""
    ui = cai.current_ui()
    if not ui.interactive:
        return "Error: no user is reachable to answer"
    if not options:
        return "Error: options must not be empty"
    answer = ui.select(question, options)
    if answer is None:
        return "Error: the user did not choose"
    return answer


@cai.tool
def choose_many(question: str, options: list[str]) -> str:
    """Ask the user to pick any number of the options; returns the chosen ones, one per line."""
    ui = cai.current_ui()
    if not ui.interactive:
        return "Error: no user is reachable to answer"
    if not options:
        return "Error: options must not be empty"
    chosen = ui.multiselect(question, options)
    if not chosen:
        return "(none chosen)"
    return "\n".join(chosen)
