"""tui.pickers: the overlays a session's commands open - messages, history,
prompts, models, sessions, pending, config, agents (with attach), tools, skills
and the system prompt - each a picker over the agent client."""
import os

from cai import usage
from cai.events import EventType
from cai.screen import Screen
from cai.screen.ansi import SGR_RESET
from cai.screen.overlays import config as overlay_config
from cai.screen.overlays.config import Setting
from cai import sessions_index
from cai.session import SessionsRegistry
from cai.wire import Wire
from cai.tui.transcript import _AttachView, _Transcript, _replay_messages
from cai.tui.client import _CONTINUE, _agent_control
from cai.tui.status import _show_agent_errors


def _open_messages(screen, client, status):
    """open the :messages overlay over a snapshot of the conversation and write
    back any edits, both over the wire. while a run is in flight the overlay
    opens read-only and follows the turn as it appends (set_messages is applied
    between turns, so an edit made now would drop what the turn added)."""
    msgs = client.get_messages()
    if not msgs:
        screen.write("[no messages yet]\n", kind=Screen.META, block=True)
        return
    context_size, prompt_tokens, sample_chars = status.ctx_snapshot()
    readonly = screen.busy
    refetch = None
    revision = None
    if readonly:
        refetch = client.get_messages
        revision = client.get_revision
    edited, _estimate, modified = screen.prompt_messages_overlay(
        msgs,
        context_size=context_size,
        prompt_tokens=prompt_tokens,
        sample_chars=sample_chars,
        refetch=refetch,
        revision=revision,
        readonly=readonly)
    if modified and not readonly:
        client.set_messages(edited)


def _open_requests(screen, requests, agent_name, title="Requests"):
    """open the :requests overlay: the HTTP requests one agent sent to the
    model server. `requests` is the TUI's RequestStore (cai.tui.requests) -
    the recent ones it kept in memory, nothing stored on disk - and
    agent_name picks whose to show: the session's own agent, or a sub-agent
    being watched."""
    def _fetch():
        return requests.summaries(agent_name)

    def _detail(id):
        return requests.detail(agent_name, id)

    if not _fetch():
        screen.write("[no requests kept for this agent]\n", kind=Screen.META, block=True)
        return
    screen.prompt_requests_overlay(_fetch, _detail, title=title)


def _history_node_label(node):
    if node["role"] == "user":
        return "user"
    if node["has_tools"]:
        return "assistant + tools"
    return "assistant"


def _history_node_color(node):
    from cai.screen.ansi import SGR_CYAN, SGR_GREEN

    if node["role"] == "user":
        return SGR_GREEN
    return SGR_CYAN


def _history_node_preview(tree, node, width, max_lines):
    from cai.screen.render import _preview_lines

    convo = []
    for message in tree.prefix_messages(node["id"]):
        if message.get("role") == "system": continue
        convo.append(message)
    head = f"{len(convo)} messages"[:width]
    lines = [head, ""]
    body = max(1, max_lines - len(lines))
    lines.extend(_preview_lines(convo, width, body))
    return lines[:max_lines]


def _open_history(screen, client, status, jobs, cfg):
    """open the :history fork view over a snapshot of the conversation. on Enter,
    rewind the live conversation to the chosen node, repaint the viewport to match,
    and - unless the snapshot ends on a final assistant reply - re-enter the
    agentic loop with a continue turn. the tree lives on the screen across opens;
    ingest() drops and rebuilds it when the conversation no longer shares its first
    turn (a clear / load / resume), so stale branches never linger."""
    from cai.history import HistoryTree
    from cai.screen.overlays.tree import _flatten_history

    msgs = client.get_messages()
    if not msgs:
        screen.write("[no history yet]\n", kind=Screen.META, block=True)
        return
    tree = screen.history_tree
    if tree is None:
        tree = HistoryTree()
        screen.history_tree = tree
    tree.ingest(msgs)
    if not tree.nodes():
        screen.write("[no history yet]\n", kind=Screen.META, block=True)
        return

    def _preview(node, width, max_lines):
        return _history_node_preview(tree, node, width, max_lines)

    def _is_head(node):
        return node["is_head"]

    sel = screen.prompt_tree_overlay(
        tree.nodes,
        label_fn=_history_node_label,
        preview_fn=_preview,
        color_fn=_history_node_color,
        is_self_fn=_is_head,
        flatten_fn=_flatten_history,
        title="history",
        hints='  j/k /:search ↵:fork ESC:cancel')
    if sel is None:
        return
    client.set_messages(tree.prefix_messages(sel))
    screen.clear_buffer()
    _replay_messages(screen, client.get_messages(), cfg)
    if tree.should_continue(sel):
        jobs.put(_CONTINUE)


def _prompt_entries(history):
    """the :prompts entries: label -> full prompt, newest first, blanks
    skipped and duplicates dropped (the newest occurrence wins). the label
    is the prompt flattened to one line - it is what the fuzzy filter
    matches against, so the whole prompt is searchable, not just the part
    that fits the row."""
    entries = {}
    for entry in history:
        label = " ".join(entry.split())
        if not label: continue
        if label in entries: continue
        entries[label] = entry
    return entries


def _open_prompts(screen):
    """open the :prompts picker - a fuzzy finder over the prompt history.
    picking an entry drops it into the input box for editing, exactly like
    the Ctrl-X recall; nothing is submitted until the user hits Enter."""
    from cai.screen.ansi import wrap_ansi

    entries = _prompt_entries(screen.prompt_history())
    if not entries:
        screen.write("[no prompt history]\n", kind=Screen.META, block=True)
        return

    def _preview(label, width, max_lines):
        return wrap_ansi(entries[label], width)[:max_lines]

    picked = screen.prompt_history_overlay(list(entries), preview_fn=_preview)
    if picked is None:
        return
    screen.set_input(entries[picked])


def _open_models(screen, client, status, registry):
    """open the :models picker and switch the agent (over the wire) to the chosen
    model. the catalogue (and prices) come from the registry, which fetches+caches
    from the provider; pins are persisted by the overlay through the same
    registry."""
    ids = registry.model_ids()
    if not ids:
        screen.write("[no models available]\n", kind=Screen.META, block=True)
        return
    picked = screen.prompt_model_overlay(ids, prices=registry.prices(), favorites=True)
    if picked:
        client.set_model(picked)
        status.set_model(picked)


def _load_session(screen, client, status, path, cfg):
    """load a .flow into the agent over the wire, then refresh the view: clear the
    viewport, replay the conversation, and follow the restored model. shared by
    :load, the :sessions picker, and the --continue/--sessions resume flags."""
    if not client.load(path):
        screen.write(f"[load error] {path}\n", kind=Screen.ERROR, block=True)
        _show_agent_errors(screen, client)
        return
    status.set_model(client.get_info().get("model", ""))
    screen.clear_buffer()
    _replay_messages(screen, client.get_messages(), cfg)
    screen.write(f"[loaded {os.path.basename(path)}]\n", kind=Screen.META, block=True)


def _session_tree_nodes(wait=0.0):
    """saved sessions as a hierarchy: each .flow declares the ids of the
    sub-agents it launched (their own '<id>.flow' stems), so a child nests under
    the parent that declared it. nodes link by id; a declared child whose own
    flow was deleted becomes a dead-end leaf rather than vanishing. served from
    the sessions index (cai.sessions_index): the picker calls this twice a
    second, and the index only stats the folder - a .flow is read once, in the
    background, and again only when it changes."""
    index = sessions_index.default()
    index.refresh(wait)
    return index.nodes()


def _open_sessions(screen, client, status, cfg):
    """open the :sessions picker (a tree of sessions and their sub-agent
    sessions) and load the chosen saved session. the rows, the search and the
    preview all come from the in-memory sessions index, so moving between
    sessions never waits on the disk; the load itself goes over the wire.
    '/' filters to the sessions whose messages hold every word typed; the
    preview reads from the session's first message (PgDn pages on). Ctrl-K
    deletes the highlighted session's .flow; a pruned dead-end has no
    transcript to delete."""
    from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET

    index = sessions_index.default()
    query = [""]             # the filter in force, for the preview's match rows

    def _fetch():
        return _session_tree_nodes()

    def _filter(pattern):
        query[0] = pattern
        return index.search(pattern)

    if not SessionsRegistry.list_sessions():
        screen.write("[no saved sessions]\n", kind=Screen.META, block=True)
        return

    def _label(node):
        name = node.get("name") or node["id"]
        if node.get("dead_end"):
            return f"{name} · pruned"
        return name

    def _color(node):
        if node.get("dead_end"):
            return SGR_DIM_GRAY
        return ""

    def _preview(node, width, max_lines):
        if node.get("path") is None:
            return [f"{SGR_DIM_GRAY}pruned - no saved transcript{SGR_RESET}"]
        return index.preview(node["path"], width, max_lines, query[0])

    def _delete(nid):
        for node in _session_tree_nodes():
            if node["id"] != nid: continue
            path = node.get("path")
            if path is None: return
            try:
                os.remove(path)
            except OSError:
                pass
            return

    # give the background read a blink to finish, so an ordinary sessions
    # folder is complete on the first frame; a huge one fills in as it lands.
    _session_tree_nodes(wait=0.15)
    sel = screen.prompt_tree_overlay(
        _fetch,
        label_fn=_label,
        preview_fn=_preview,
        color_fn=_color,
        action_fn=_delete,
        filter_fn=_filter,
        preview_from_top=True,
        title="sessions",
        hints='  j/k PgDn/Up:preview /:search all messages ↵:resume ^K:delete ESC:cancel',
    )
    if sel is None:
        return
    for node in _session_tree_nodes():
        if node["id"] != sel: continue
        path = node.get("path")
        if path is None:
            return
        _load_session(screen, client, status, path, cfg)
        return


def _pending_nodes(steers, queued):
    """the :pending rows: steers first (a run folds them in at its next turn
    boundary, ahead of anything queued), then the queued prompts in run order.
    ids are kind + text + its occurrence number, so removing one row never
    renames another and the overlay keeps its cursor on the same text."""
    nodes = []
    seen = {}

    def add(kind, index, text):
        label = text
        if label is _CONTINUE:
            label = "(continue)"
        key = (kind, label)
        n = seen.get(key, 0)
        seen[key] = n + 1
        node = {}
        node["id"] = f"{kind}:{n}:{label}"
        node["parent"] = None
        node["kind"] = kind
        node["index"] = index
        node["item"] = text
        node["text"] = label
        nodes.append(node)

    for i, text in enumerate(steers):
        add("steer", i, text)
    for i, item in enumerate(queued):
        if item is None: continue
        add("queued", i, item)
    return nodes


def _open_pending(screen, client, jobs, pending):
    """open the :pending view: the steers the agent holds and the prompts
    queued locally behind the running turn, oldest first. Ctrl-K removes the
    highlighted one (a steer over the wire, a prompt from the local queue);
    Enter/Esc close. opens only when there is something to show."""
    from cai.screen.ansi import wrap_ansi, SGR_DIM_GRAY, SGR_AZURE_ON_DGRAY, SGR_RESET

    def _fetch():
        return _pending_nodes(client.get_steer(), jobs.snapshot())

    if not _fetch():
        screen.write("[nothing pending]\n", kind=Screen.META, block=True)
        return

    def _label(node):
        head = node["text"].strip().splitlines()
        first = ""
        if head:
            first = head[0]
        tag = node["kind"]
        if not node.get("present", True):
            return f"{tag}  {first} · done"
        return f"{tag}  {first}"

    def _color(node):
        if not node.get("present", True):
            return SGR_DIM_GRAY
        if node["kind"] == "steer":
            return SGR_AZURE_ON_DGRAY
        return ""

    def _preview(node, width, max_lines):
        return wrap_ansi(node["text"], width)[:max_lines]

    def _remove(nid):
        for node in _fetch():
            if node["id"] != nid: continue
            if node["kind"] == "steer":
                client.remove_steer(node["index"], node["item"])
                return
            if jobs.remove(node["index"], node["item"]) and node["item"] is not _CONTINUE:
                pending.user_removed()
            return

    screen.prompt_tree_overlay(
        _fetch,
        label_fn=_label,
        preview_fn=_preview,
        color_fn=_color,
        action_fn=_remove,
        title="pending",
        hints='  j/k PgUp/Dn:preview /:search ^K:remove ESC:close',
    )


def _bool_setting(label, obj, attr):
    return Setting(label=label,
                   kind=overlay_config.BOOL,
                   get=lambda: getattr(obj, attr),
                   set=lambda v: setattr(obj, attr, bool(v)))


def _int_setting(label, obj, attr):
    """an integer field. empty input clears to 0; non-numbers report an error
    the overlay shows instead of writing. the overlay's undo replays the native
    int through set(), which str()/int() round-trips cleanly."""
    def write(value):
        text = str(value).strip()
        if text == "":
            setattr(obj, attr, 0)
            return None
        try:
            setattr(obj, attr, int(text))
        except ValueError:
            return "not a number"
        return None
    return Setting(label=label,
                   kind=overlay_config.INT,
                   get=lambda: getattr(obj, attr),
                   set=write)


def _list_setting(label, obj, attr):
    """a list of names edited as a comma-separated string. set() accepts either
    the edited text or a native list (the overlay's undo replays the displayed
    string, which splits back to the same list)."""
    def write(value):
        if isinstance(value, (list, tuple)):
            setattr(obj, attr, list(value))
            return None
        names = []
        for part in str(value).split(","):
            part = part.strip()
            if part:
                names.append(part)
        setattr(obj, attr, names)
        return None
    return Setting(label=label,
                   kind=overlay_config.STRING,
                   get=lambda: ", ".join(getattr(obj, attr) or []),
                   set=write)


def _open_config(screen, client, cfg):
    """the :config overlay - edit the env's live Settings (cai.settings) in
    place. edits apply to this session only; permanent config lives in init.py."""
    settings = []
    settings.append(_bool_setting("show reasoning", cfg, "show_reasoning"))
    settings.append(_int_setting("tool result max chars", cfg, "tool_result_max_chars"))
    settings.append(_bool_setting("auto save sessions", cfg, "auto_save_sessions"))
    settings.append(_int_setting("max sessions mb", cfg, "max_sessions_mb"))
    settings.append(_list_setting("skills", cfg, "skills"))
    settings.append(_list_setting("tools", cfg, "tools"))
    screen.prompt_config_overlay(settings)


def _attach_agent(screen, node, cfg, stop_fn, requests, limit_fn=None):
    """watch one live agent from the :agents view: attach a read-only wire to
    its socket and mirror its conversation full-screen - the stored messages
    first (get_messages over a short-lived control connection), then the run's
    EVENT broadcast streamed live (a UnixWiredAgent broadcasts to every
    attached wire, and a sub-agent's owner thread keeps its own wire, so this
    one may come and go freely without stopping anything). nothing is ever
    sent on the wire - no SUBMIT/STEER/INTERRUPT, and PROMPT broadcasts are
    ignored (the owner answers them) - so the watched agent cannot be driven
    from here. the snapshot is fetched after the stream attaches, so a message
    completing in between may render twice - a duplicated tool line at worst,
    never a lost one. Ctrl-L hands off to the messages overlay (the overlays
    each own the terminal, so it opens here, between attach entries) and then
    re-attaches - a fresh connect and snapshot, same as the first entry.
    returns False when the socket is unreachable on the first attempt (the
    agent just finished), so the caller falls back to the scrollback attach
    view over the autosaved flow; an agent that dies during the messages
    overlay ends the dive back at the tree instead. limit_fn(model) gives the
    context window for the status row's ctx readout, whose token count starts
    from get_info and follows the USAGE events streamed while attached."""
    from cai.agents_registry import AgentsRegistry

    name = node["id"]
    title = node.get("name") or name
    attached = False
    limit = 0
    if limit_fn is not None:
        limit = limit_fn(node.get("model", ""))
    tokens = {}
    tokens["total"] = 0

    def _ctx():
        if not limit:
            return ''
        return usage.format_ctx(tokens["total"], limit)
    while True:
        try:
            channel = AgentsRegistry.connect(name)
        except OSError:
            return attached
        stream = Wire(channel)
        snapshot = _agent_control(name, "get_messages")
        info = _agent_control(name, "get_info") or {}
        tokens["total"] = info.get("tokens", 0)
        view = _AttachView(screen.cols)
        _replay_messages(view, snapshot or [], cfg)
        transcript = _Transcript(view, cfg)

        def _drain():
            try:
                messages = stream.recv()
            except OSError:
                messages = None
            if messages is None:
                transcript.note(f"[{title} finished]\n", Screen.META)
                return False
            for msg in messages:
                if msg.get("type") != Wire.EVENT: continue
                event = Wire.event_from_dict(msg["event"])
                if event.type == EventType.USAGE:
                    tokens["total"] = usage.total_tokens(event.usage)
                transcript.event(event)
            return True

        def _kill():
            stop_fn(name)

        try:
            action = screen.prompt_attach_overlay(view,
                                                  title=title,
                                                  watch=channel,
                                                  drain_fn=_drain,
                                                  kill_fn=_kill,
                                                  ctx_fn=_ctx)
        finally:
            try:
                channel.close()
            except OSError:
                pass
        attached = True
        if action == "requests":
            _open_requests(screen, requests, name, f"Requests · {title}")
            continue
        if action != "messages":
            return True
        messages = _agent_control(name, "get_messages")
        if messages is None:
            messages = snapshot or []
        screen.prompt_messages_overlay(messages)


_PREVIEW_MESSAGES = 40


def _conversation_tail(messages, width, max_lines, cfg=None):
    """the last max_lines rows of a conversation painted the way the attach
    view paints it - what the :agents tree previews, so the pane shows exactly
    what Enter opens. only the newest messages are replayed (rows above the
    tail would be cut anyway), starting on a user or assistant turn so a tool
    reply never opens the preview without its call."""
    tail = messages
    if len(messages) > _PREVIEW_MESSAGES:
        tail = messages[-_PREVIEW_MESSAGES:]
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    view = _AttachView(width)
    _replay_messages(view, tail, cfg)
    total = view.line_count()
    return view.get_lines(max(0, total - max_lines), max_lines)


def _open_agents(screen, client, cfg, requests, status=None):
    """open the live sub-agents tree, built solely from the agents sockets. each
    live agent answers get_info (name, model, the ids of its children); the tree
    is linked from those children lists, and a child with no live socket of its
    own (finished, torn down) still shows as a leaf. preview reads a live agent's
    conversation over its socket; Enter attaches to it read-only (a finished one
    opens its stored transcript instead); Ctrl-K interrupts a live sub-agent.
    the TUI's own agent is the root, marked self - Enter on it just returns to
    the conversation, which is that very view."""
    from cai.agents_registry import AgentsRegistry
    from cai.screen.ansi import SGR_DIM_GRAY, SGR_RESET

    self_name = client.get_info().get("name", "")

    def _nodes():
        nodes = {}
        for name in AgentsRegistry.list_names():
            info = _agent_control(name, "get_info")
            if info is None: continue
            node = {}
            node["id"] = name
            node["parent"] = None
            node["name"] = info.get("name") or name
            node["model"] = info.get("model", "")
            node["children_ids"] = info.get("children") or []
            node["status"] = "running"
            node["live"] = True
            nodes[name] = node
        for node in list(nodes.values()):
            for cid in node["children_ids"]:
                if cid in nodes: continue
                leaf = {}
                leaf["id"] = cid
                leaf["parent"] = None
                leaf["name"] = cid
                leaf["model"] = ""
                leaf["children_ids"] = []
                leaf["status"] = "done"
                leaf["live"] = False
                nodes[cid] = leaf
        for node in nodes.values():
            for cid in node["children_ids"]:
                child = nodes.get(cid)
                if child is None: continue
                child["parent"] = node["id"]
        root = nodes.get(self_name)
        if root is not None:
            root["status"] = "idle"
            if screen.busy:
                root["status"] = "running"
        for node in nodes.values():
            node["children_count"] = len(node["children_ids"])
        return list(nodes.values())

    def _messages_for(node):
        # prefer the live transcript over the socket; once an agent finishes its
        # socket is gone, so fall back to the '<id>.flow' it autosaved (its id is
        # the flow stem). non-system messages only - the leading composed prompt
        # is not worth previewing.
        if node.get("live"):
            messages = _agent_control(node["id"], "get_messages")
            if messages is not None:
                return messages
        try:
            payload = SessionsRegistry.read_flow(SessionsRegistry.session_path(node["id"]))
        except (OSError, ValueError):
            return None
        convo = []
        for message in (payload.get("messages") or []):
            if message.get("role") == "system": continue
            convo.append(message)
        return convo

    def _preview(node, width, max_lines):
        status = node.get("status", "")
        if not node.get("present", True):
            status = "finished"
        lines = [status[:width], ""]
        messages = _messages_for(node)
        if messages is None:
            lines.append(f"{SGR_DIM_GRAY}no transcript{SGR_RESET}")
            return lines[:max_lines]
        body = max(1, max_lines - len(lines))
        lines.extend(_conversation_tail(messages, width, body, cfg))
        return lines[:max_lines]

    def _stop(name):
        if name == self_name:
            return
        try:
            channel = AgentsRegistry.connect(name)
        except OSError:
            return
        try:
            Wire(channel).send_interrupt()
        finally:
            channel.close()

    # Enter dives into the selected agent, ESC backs out one level: the attach
    # view (live-streaming for a running agent, pure scrollback for a finished
    # one) returns to the tree, the tree to the conversation.
    while True:
        sel = screen.prompt_agents_overlay(_nodes, _preview, stop_fn=_stop,
                                           self_id=self_name)
        if sel is None:
            return
        chosen = None
        for node in _nodes():
            if node["id"] != sel: continue
            chosen = node
            break
        if chosen is None:
            return
        if chosen["id"] == self_name:
            return
        limit_fn = None
        if status is not None:
            limit_fn = status.limit_for
        if chosen.get("live") and _attach_agent(screen, chosen, cfg, _stop, requests, limit_fn):
            continue
        messages = _messages_for(chosen)
        if messages is None:
            screen.write("[no transcript for this agent]\n", kind=Screen.META, block=True)
            return
        view = _AttachView(screen.cols)
        _replay_messages(view, messages, cfg)
        while True:
            action = screen.prompt_attach_overlay(view,
                                                  title=chosen.get("name") or chosen["id"])
            if action == "requests":
                title = chosen.get("name") or chosen["id"]
                _open_requests(screen, requests, chosen["id"], f"Requests · {title}")
                continue
            if action != "messages":
                break
            screen.prompt_messages_overlay(messages)


def _tool_label(name):
    """the origin column for a tool entry: the MCP server name (the part before
    '__'), or 'tool' for an unprefixed function tool."""
    if "__" in name:
        return name.split("__", 1)[0]
    return "tool"


def _open_tools(screen, client, cfg):
    """open the :tools overlay and apply the new selection, all over the wire.
    get_available_tools already unions the catalogue with the agent's own
    registered tools, so it is the full selectable set."""
    active = set(client.get_selected_tools())
    available = set(client.get_available_tools())
    if not available:
        screen.write("[no tools available]\n", kind=Screen.META, block=True)
        return
    entries = []
    for name in sorted(available):
        entries.append((name, _tool_label(name)))
    new_selected = screen.prompt_tools_overlay(entries, active)
    client.set_selected_tools(sorted(new_selected))
    _show_agent_errors(screen, client)


def _open_skills(screen, client, cfg):
    """open the :skills overlay and apply the new selection, all over the wire."""
    available = set(client.get_available_skills())
    active = set(client.get_selected_skills())
    if not available:
        screen.write("[no skills available]\n", kind=Screen.META, block=True)
        return
    new_active = screen.prompt_skills_overlay(sorted(available), active)
    client.set_selected_skills(sorted(new_active))
    _show_agent_errors(screen, client)


def _system_prompt_nodes(composed, base):
    nodes = []

    readonly = {}
    readonly["id"] = "readonly"
    readonly["parent"] = None
    readonly["label"] = "full prompt (read-only)"
    readonly["text"] = composed
    readonly["empty"] = "(no system prompt set)"
    nodes.append(readonly)

    edit = {}
    edit["id"] = "base"
    edit["parent"] = None
    edit["label"] = "base (edit — in memory)"
    edit["text"] = base
    edit["empty"] = "(empty)"
    nodes.append(edit)

    return nodes


def _system_preview(node, width, max_lines):
    from cai.screen.ansi import wrap_ansi, SGR_RESET, SGR_DIM_GRAY

    text = node["text"]
    if not text:
        return [f"{SGR_DIM_GRAY}{node['empty']}{SGR_RESET}"]
    return wrap_ansi(text, width)[:max_lines]


def _open_system(screen, client):
    """`:system` overlay. two options: the full composed prompt (read-only), and
    the agent's system-prompt base, editable in nvim. saving the base pushes it
    to the live agent (which recomposes the prompt with the current skills); the
    change is in memory only and never touches disk."""
    info = client.get_info()
    composed = info.get("system_prompt") or ""
    base = info.get("system_prompt_base") or ""
    nodes = _system_prompt_nodes(composed, base)

    def _fetch():
        return nodes

    def _label(node):
        return node["label"]

    # loop so closing the editor returns to the system view, not the main
    # conversation; only ESC at the overlay exits.
    while True:
        sel = screen.prompt_tree_overlay(
            _fetch,
            label_fn=_label,
            preview_fn=_system_preview,
            title="system prompt",
            hints='  j/k ↵:open ESC:cancel')
        if sel is None:
            return
        if sel == "readonly":
            if not composed:
                screen.write("[no system prompt set]\n", kind=Screen.META, block=True)
                continue
            screen.view_in_editor(composed, suffix='.md')
            continue
        edited = screen.edit_in_editor(base, suffix='.md')
        if edited is None:
            continue
        client.set_system_prompt_base(edited)
        base = edited
        info = client.get_info()
        composed = info.get("system_prompt") or ""
        base = info.get("system_prompt_base") or ""
        nodes = _system_prompt_nodes(composed, base)
