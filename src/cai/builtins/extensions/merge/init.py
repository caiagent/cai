"""init.py - bundled extension: merging one TUI tab into another (:merge N).

`:merge N` summarises what tab N did beyond what the tab in front already
has, lets the user review the summary in the editor, delivers it into the tab
in front, then closes tab N. the grammar is git's: the tab you are on is the
one merged into, and it stays in front. merging is one way - a tab is merged
once and is gone.

It runs start to finish in the command, on the main thread: the summary
streams into the conversation view as the model writes it, Ctrl-C cancels it,
and the review opens the moment it is done. nothing is left running behind
the prompt.

It ships with cai (builtins/extensions/) but is an ordinary extension, written
against what any `:`-command gets: ctx.tab and ctx.tabs for the tabs, each
tab's client (the wire) and screen, and cai.Run for the model call. To change how a merge
works, install a bundle that registers its own `merge` - a later registration
of a name wins."""
import logging

import cai
from cai.origin import origin_label
from cai.origin import origin_of
from cai.origin import tab_origin


log = logging.getLogger("cai")


_MERGE_SYSTEM = """You write handoff summaries between two parallel work sessions that started from the same conversation.

You are given the conversation both sessions share, then the work one of them did afterwards. Write a summary of that later work for the other session, which has seen the shared part and nothing after it.

Cover what matters for continuing the work: what was done and decided, what was found out, which files were created or changed (exact paths), commands or results worth knowing, and what is still open or failed. Be specific and concrete, quote exact names and values, and leave out the back-and-forth of how it got there. Do not repeat what the shared conversation already says. Write the summary itself and nothing else - no preamble, no closing remarks."""


# how much of one tool result the summariser reads: little of the shared part
# (it only sets the scene), more of the work being summarised.
_MERGE_SHARED_TOOL_CHARS = 400


_MERGE_WORK_TOOL_CHARS = 4000


# the note at the top of the summary while it is open for review; stripped
# before the summary is sent.
_MERGE_REVIEW_NOTE = "<!-- {what}: edit freely, then save and quit. empty the file to cancel. this note is removed. -->\n\n"


def _note(screen, text):
    """a one-line answer in a tab's conversation view."""
    from cai.screen import Screen

    screen.write(f"[{text}]\n", kind=Screen.META, block=True)


class _Progress:
    """the merge's one block in the conversation view, rewritten in place:
    the summary as it streams, then the line saying how the merge ended."""

    def __init__(self, screen, heading):
        from cai.screen import Screen

        self._screen = screen
        self._meta = Screen.META
        self._error = Screen.ERROR
        self._heading = heading
        self._block = screen.write_block(f"[{heading}]", kind=self._meta)

    def show(self, text):
        """the heading over what the model has written so far."""
        self._screen.replace_block(self._block, f"[{self._heading}]\n{text}", kind=self._meta)

    def end(self, text, error=False):
        """fold the block to the one line that stays in the view."""
        kind = self._meta
        if error:
            kind = self._error
        self._screen.replace_block(self._block, f"[{text}]", kind=kind)


def _settled(messages):
    """the conversation cut back to before an assistant message whose tool
    calls have not all been answered - what an interrupted run can leave at
    the end. half a tool exchange is not work to summarise."""
    messages = list(messages)
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.get("role") == "tool": continue
        if message.get("role") != "assistant": break
        calls = message.get("tool_calls") or []
        if len(calls) > len(messages) - 1 - i:
            return messages[:i]
        break
    return messages


def _merge_point(source, target):
    """where source stops being the conversation target already has: the
    length of their common prefix. a branched tab starts from a copy of the
    conversation, so the shared messages compare equal by content."""
    shared = 0
    for mine, theirs in zip(source, target):
        if mine != theirs: break
        shared += 1
    return shared


def _merge_transcript(messages, tool_chars):
    """messages as plain text for the summariser - a transcript it reads,
    not a conversation it is in, so no provider rule about tool calls and
    their results applies. a long tool result is cut to tool_chars."""
    lines = []
    for message in messages:
        role = message.get("role")
        if role == "system": continue
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = "[non-text content]"
        if role == "tool":
            if len(content) > tool_chars:
                content = content[:tool_chars] + f"… [{len(content) - tool_chars} more chars]"
            lines.append(f"TOOL RESULT:\n{content}")
            continue
        if role == "user":
            origin = origin_of(message)
            if origin is not None:
                lines.append(f"REPORT FROM {origin_label(origin)}:\n{content}")
                continue
            lines.append(f"USER:\n{content}")
            continue
        if content.strip():
            lines.append(f"ASSISTANT:\n{content}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = str(function.get("arguments") or "")
            if len(arguments) > 600:
                arguments = arguments[:600] + "…"
            lines.append(f"ASSISTANT CALLED TOOL {function.get('name')}: {arguments}")
    return "\n\n".join(lines)


def _merge_request(shared, work):
    """the prompt asking the model for the summary of `work`, the part of a
    conversation that follows `shared`."""
    shared_text = _merge_transcript(shared, _MERGE_SHARED_TOOL_CHARS)
    if not shared_text:
        shared_text = "(nothing - the sessions share no history)"
    return (f"<shared_conversation>\n{shared_text}\n</shared_conversation>\n\n"
            f"<later_work>\n{_merge_transcript(work, _MERGE_WORK_TOOL_CHARS)}\n</later_work>\n\n"
            "Write the handoff summary of the later work.")


def _merge_worth_sending(work):
    """whether `work` holds anything the model did - an answer, a tool call -
    rather than nothing, or only prompts nobody answered yet."""
    for message in work:
        if message.get("role") == "assistant":
            return True
    return False


def _summarise(request, model):
    """one model call of its own for the summary, as a stream of
    (event type, text) - the model's reasoning, then its answer - so the
    caller can show it being written. a throwaway agent on an empty
    Environment - no tools, no skills, none of the session's hooks - so
    neither tab's conversation takes part in it. closing the generator (or an
    exception thrown through it, like Ctrl-C) closes the agent."""
    prompt = {}
    prompt["role"] = "user"
    prompt["content"] = request
    run = cai.Run(messages=[prompt],
                  model=model,
                  env=cai.Environment(),
                  system_prompt=_MERGE_SYSTEM)
    try:
        for event in run:
            if not event.text: continue
            if event.type not in (cai.EventType.CONTENT, cai.EventType.REASONING): continue
            yield str(event.type), event.text
    except BaseException:
        # cancelled or failed part way: stop the request, not just the read.
        run.interrupt.set()
        raise
    finally:
        run.close()


def _write_summary(progress, request, model):
    """stream the summary into the view and return it. reasoning is shown
    until the answer starts, then the answer alone. KeyboardInterrupt (Ctrl-C)
    and a failed call pass through to the caller."""
    reasoning = []
    content = []
    for kind, text in _summarise(request, model):
        if kind == cai.EventType.CONTENT:
            content.append(text)
            progress.show("".join(content))
            continue
        if content: continue
        reasoning.append(text)
        progress.show("".join(reasoning))
    return "".join(content).strip()


@cai.command(help="summarise another tab's work (its number) into this one, then close that tab",
             takes_arg=True)
def merge(ctx: cai.CommandContext):
    """:merge N. refused while tab N is running; this tab may be.

    synchronous: the command returns when the merge is done, cancelled or
    failed, and a merge that does not go through leaves tab N open. the
    summary is delivered like a prompt, as a steer tagged with tab N's
    session (cai.origin): this tab folds it into a run in flight at the next
    step, and starts a turn on it when idle."""
    tabs = ctx.tabs
    target = ctx.tab
    screen = target.screen
    source = None
    if ctx.args.isdigit():
        source = tabs.at(int(ctx.args))
    if source is None:
        _note(screen, ":merge needs the number of an open tab")
        return
    if source == target:
        _note(screen, ":merge — that is this tab")
        return
    number = source.number
    if source.busy:
        _note(screen, f":merge — tab {number} is still running; wait for it or interrupt it first")
        return

    messages = _settled(source.client.get_messages())
    start = _merge_point(messages, target.client.get_messages())
    work = messages[start:]
    if not _merge_worth_sending(work):
        _note(screen, f"nothing new to merge from tab {number}")
        return
    request = _merge_request(messages[:start], work)
    info = source.client.get_info()
    name = info.get("name")
    model = info.get("model") or None

    progress = _Progress(screen, f"merging tab {number} — writing the summary, Ctrl-C cancels")
    try:
        summary = _write_summary(progress, request, model)
    except KeyboardInterrupt:
        progress.end(f"merge of tab {number} cancelled")
        return
    except Exception as e:
        log.exception("merge summary failed")
        progress.end(f"error: merge summary failed: {type(e).__name__}: {e}", error=True)
        return
    if not summary:
        progress.end("merge cancelled — the model returned no summary")
        return

    note = _MERGE_REVIEW_NOTE.format(what=f"summary of tab {number} to merge into this tab")
    edited = screen.edit_in_editor(note + summary + "\n", suffix=".md")
    if edited is not None:
        summary = edited.replace(note.strip(), "", 1).strip()
    if not summary:
        progress.end(f"merge of tab {number} cancelled")
        return
    # the same path a sub-agent's report takes: this tab's transcript paints
    # the summary when its agent folds it in.
    target.client.steer(summary, origin=tab_origin(name))
    tabs.close(source)
    progress.end(f"merged tab {number}")
