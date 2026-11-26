"""skills: load skill files into a SkillsRegistry that layers prompt fragments
and tools onto an existing ToolsRegistry.

A skill is a markdown file <name>.md, resolved from the Environment's skill
dirs - each extension's skills/ dir first, then the builtins shipped with cai
(builtins/skills/ inside the package). Everything before the first '---' line is
a header of 'key: value' lines; everything after is the prompt body. Header
fields:

  name:   <skill name>             the added first field, naming the skill
  tools:  <tool>, <tool>, ...      MCP tool refs to expose ('<mcp>__<tool>')
  skills: <skill>, <skill>, ...    other skills to activate first (recursive)

Activating a skill unions its tools into the ToolsRegistry it was built against
and appends its prompt body to the combined system prompt. `skills:` are
activated first (foundation-first), and each skill loads at most once. The
install-wide catalogue (which skills exist) is the env's:
env.available_skills() lists it.

A skill body may carry {{name}} slots - holes filled fresh on every read of
system_prompt (once per turn), so a skill can push live state to the model
instead of making it act to fetch it. A few names are builtin: {{tools}} (the
currently selected tools' signatures), {{allowed_paths}} (the
CAI_ALLOWED_PATHS grants, one per line, "(none)" without any), and
{{python_sandbox}} / {{python_sandbox_writes}} (what the python tool's sandbox
allows under the current `python_sandbox` setting); any other name
resolves to a filler registered via the cai.slot decorator, called with a
SlotContext:

    @cai.slot
    def notes(ctx):
        return read_store()          # -> fills {{memory__notes}}

A filler registered while an extension loads is namespaced
'<extension>__<name>' like a tool. It should be fast - it runs every turn. An
unknown slot fills empty (warned); a filler that raises fills with an error
marker (logged) - one bad slot never breaks the turn."""
from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass

from cai import config
from cai import paths
from cai.environment import Environment


log = logging.getLogger("cai")


# a {{name}} hole in a skill body: a builtin ({{tools}}, {{allowed_paths}},
# {{python_sandbox}}, {{python_sandbox_writes}}) or a cai.slot filler.
_SLOT_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


@dataclass(frozen=True)
class SlotContext:
    """passed to every slot filler. `agent` is the live Agent whose prompt is
    being built (messages, model, scratch, env are all reachable from it), or
    None for a registry built without one; `skill` is the name of the skill
    whose body carries the slot."""
    agent: object
    skill: str


def slot(fn):
    """decorator: register a skill-body slot filler on the current Environment,
    e.g.

        @cai.slot
        def notes(ctx):
            \"\"\"fills {{notes}} ({{<extension>__notes}} in a bundle).\"\"\"
            return "\\n".join(read_store())

    the slot's name is the function's __name__ (namespaced
    '<extension>__<name>' when an extension is being loaded). it lands on the
    env being load()ed - else the process default - so once Environment.load()
    imports the extensions, any skill body on that env can carry the
    {{name}} hole. called with a SlotContext on every system_prompt read."""
    Environment.target().register_slot(fn)
    return fn


@dataclass
class Skill:
    name: str
    tools: list
    skills: list
    body: str


def _allowed_paths_slot():
    """fills {{allowed_paths}}: the working directory by its real path, then
    the CAI_ALLOWED_PATHS grants labelled as extras on top of it, then the
    CAI_DISALLOWED_PATHS denies - realpath'd, the same lists safe_path and the
    python jail honour. naming the cwd keeps the model from mistaking a grant
    for its working directory."""
    lines = [f"working directory (relative paths resolve here): {os.path.realpath(os.getcwd())}"]
    granted = paths.allowed_paths()
    if not granted:
        lines.append("extra granted paths: (none)")
    else:
        lines.append("extra granted paths, in addition to the working directory (not your cwd):")
        for root in granted:
            lines.append(f"- {root}")
    denied = paths.disallowed_paths()
    if not denied:
        lines.append("denied paths: (none)")
    else:
        lines.append("denied paths (never touch, even inside the above):")
        for root in denied:
            lines.append(f"- {root}")
    return "\n".join(lines)


# what the python-read-only skill tells the model the sandbox allows. the
# skill's rules must match what the tool enforces, and that depends on the
# `python_sandbox` setting - hence a slot, filled on every read.
_PYTHON_SANDBOX_ON = """\
- Read/List only paths listed here and scratch (`os.environ['CAI_SCRATCH']`)
  (a directory covers its subtree, a file just itself):

{allowed_paths}

- Running programs is allowed (`subprocess` & co). A spawned program runs
  inside the same sandbox: it sees the same files, writes only where you can
  write, and has no network. `ps`-style tools don't work (no `/proc`);
  a program needing a temp dir gets scratch via `TMPDIR`.
- No network.
- Writes land in scratch ONLY - use it to pass state between calls. If a task
  needs to change the tree, ask to switch to the `python` skill instead of
  attempting it.
    - To change or perform any action outside of the python environment - use
      the provided dedicated tools."""

_PYTHON_SANDBOX_OFF = """\
- No sandbox: a snippet runs as an ordinary process of the user's. It may read
  and write any file the user can, use the network, and run any program
  (`subprocess` & co) - nothing is confined.
- Relative paths resolve in the working directory: {cwd}
- Scratch (`os.environ['CAI_SCRATCH']`, also `TMPDIR`) is there for
  intermediates and for passing state between calls."""

# the same for the python (read-write) skill's own paragraph.
_PYTHON_WRITES_ON = """\
Adds write access to `python-read-only`: a snippet - and any program it runs -
may now WRITE under the current-working-directory and the granted paths as
well as scratch, the same policy the fs tools enforce. Nothing else is
writable, and nothing else exists."""

_PYTHON_WRITES_OFF = """\
The sandbox is off, so this skill adds nothing to `python-read-only`: a
snippet - and any program it runs - may already write anywhere the user can."""


def _python_sandbox_is_off():
    """whether the python tool runs with no sandbox at all (the
    `python_sandbox: "none"` setting - see cai.pytool)."""
    return config.load_optional("python_sandbox", "kernel") == "none"


def _python_sandbox_slot():
    """fills {{python_sandbox}}: the python-read-only skill's rules about what
    a snippet may read, write and reach - the sandbox's when it is on, a plain
    statement that there is none when it is off."""
    if _python_sandbox_is_off():
        return _PYTHON_SANDBOX_OFF.format(cwd=os.path.realpath(os.getcwd()))
    listed = []
    for line in _allowed_paths_slot().split("\n"):
        listed.append("  " + line)
    return _PYTHON_SANDBOX_ON.format(allowed_paths="\n".join(listed))


def _python_sandbox_writes_slot():
    """fills {{python_sandbox_writes}}: what the python skill adds on top of
    python-read-only - wider writes under the sandbox, nothing without one."""
    if _python_sandbox_is_off():
        return _PYTHON_WRITES_OFF
    return _PYTHON_WRITES_ON


def _skill_path(name, dirs):
    """resolve <name>.md to a source file, searching `dirs` in order (the env's
    skill dirs: extensions first, then the bundled builtins). None when no dir
    has it."""
    filename = name + ".md"
    for directory in dirs:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


def _split_csv(value):
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part: continue
        items.append(part)
    return items


def _parse_skill(text):
    """split a skill file into its header fields and prompt body. a file with no
    '---' line is all body, declaring no tools and no uses."""
    lines = text.splitlines()
    sep_index = None
    for i, line in enumerate(lines):
        if line.strip() != "---": continue
        sep_index = i
        break

    header = {}
    body = text
    if sep_index is not None:
        body = "\n".join(lines[sep_index + 1:])
        for line in lines[:sep_index]:
            if ":" not in line: continue
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()

    return Skill(name=header.get("name", ""),
                 tools=_split_csv(header.get("tools", "")),
                 skills=_split_csv(header.get("skills", "")),
                 body=body)


def _read_skill(name, dirs):
    path = _skill_path(name, dirs)
    if path is None:
        return None
    with open(path) as f:
        text = f.read()
    return _parse_skill(text)


class SkillsRegistry:
    """The skills activated for an Agent/Run: their combined prompt plus the
    tools they pull into the ToolsRegistry they were built against. Like
    ToolsRegistry, but it always works on a ToolsRegistry passed in - activating a
    skill extends that same registry object in place. Skill files resolve
    against the env's skill dirs."""

    def __init__(self, tools_registry, env=None, agent=None):
        self.env = env or Environment.default()
        self.tools_registry = tools_registry
        # the Agent whose prompt this registry feeds, handed to slot fillers
        # via SlotContext. None for a registry built without one (tests,
        # embedders) - fillers must tolerate that.
        self.agent = agent
        # the active skill names, in activation order: this registry is the
        # single source of truth for which skills are on. dependency skills
        # (pulled via a `skills:` header) and unresolved names are kept too, so
        # names() round-trips through set_skills.
        self._names = []
        # name -> Skill for the resolved skills, foundation-first by insertion
        # (a skill is recorded after its dependencies), for the combined prompt
        # and for knowing which tools to drop on remove().
        self._skills = {}

    @classmethod
    def for_skills(cls, skills=None, *, tools_registry, env=None, agent=None):
        """activate `skills` (recursively, via each skill's `skills:` header)
        against `tools_registry`, which is extended in place with every tool the
        skills declare. returns the SkillsRegistry holding the activated skills."""
        registry = cls(tools_registry, env, agent)
        if not skills:
            return registry
        for name in skills:
            registry.add(name)
        return registry

    def names(self):
        """the active skill names, in activation order. includes dependency
        skills and names that resolved to no file, so the list round-trips
        through set_skills."""
        return list(self._names)

    def tools_by_skill(self):
        """{skill name: [the tools its header declares]} for the active
        skills, in activation order. a name that resolved to no file maps
        to an empty list."""
        grouped = {}
        for name in self._names:
            skill = self._skills.get(name)
            if skill is None:
                grouped[name] = []
                continue
            grouped[name] = list(skill.tools)
        return grouped

    def add(self, name):
        """activate a skill (and, first, the skills it builds on), unioning its
        tools into the tool registry. an already-active name is a no-op."""
        if name in self._names:
            return
        self._names.append(name)
        self._activate(name)

    def _activate(self, name):
        """register a skill's tools and record its prompt, after first activating
        the skills it builds on. an unresolved name is warned and skipped, but
        stays in names() (mirroring set_skills's input)."""
        skill = _read_skill(name, self.env.skill_dirs())
        if skill is None:
            self.tools_registry.report_error(f"skill {name!r} not found")
            return
        # foundation first: activate everything this skill builds on, then it.
        for used in skill.skills:
            if used in self._names: continue
            self._names.append(used)
            self._activate(used)
        # select the skill's tools, registering an MCP tool on demand. a tool
        # already registered (e.g. an Agent's sub-agent tools) is just selected.
        for tool_name in skill.tools:
            try:
                self.tools_registry.select(tool_name)
            except ValueError as e:
                self.tools_registry.report_error(f"skill {name!r}: bad tool {tool_name!r}: {e}")
        self._skills[name] = skill

    def remove(self, name):
        """deactivate a skill: drop its name, its prompt, and deselect every tool
        it declared - even one also selected by hand or shared with another skill
        (kept deliberately simple). the tools stay registered; dependency skills
        it pulled in stay active."""
        if name in self._names:
            self._names.remove(name)
        skill = self._skills.pop(name, None)
        if skill is None:
            return
        for tool_name in skill.tools:
            self.tools_registry.deselect(tool_name)

    @property
    def system_prompt(self):
        """the activated skills' prompt bodies, foundation-first, slots filled,
        joined - or None when no skill contributed a prompt. each body is led by
        a 'skill: <name>' line, so the model knows the registry name it grants a
        sub-agent by. slots resolve fresh on every read (the loop reads once per
        model call), so a filler's latest state reaches the model without any
        tool call."""
        parts = []
        for name, skill in self._skills.items():
            body = skill.body.strip()
            if not body: continue
            parts.append(f"skill: {name}\n" + self._fill_slots(body, name))
        if not parts:
            return None
        return "\n\n".join(parts)

    def _fill_slots(self, body, skill_name):
        """replace every {{name}} hole in one skill's body: {{tools}} with the
        currently selected tools' signatures, {{allowed_paths}} with the
        CAI_ALLOWED_PATHS grants, {{python_sandbox}} and
        {{python_sandbox_writes}} with the python tool's current rules, anything else with its cai.slot filler's
        result. an unknown name fills empty (warned); a filler that
        raises fills with an error marker (logged), so one bad slot never
        breaks the turn. a multi-line result inherits the indentation of the
        line carrying the hole, so a list slotted into a markdown bullet stays
        inside the bullet."""
        def fill(match):
            name = match.group(1)
            line_start = body.rfind("\n", 0, match.start()) + 1
            indent = body[line_start:match.start()]
            if indent.strip():
                indent = ""
            return _resolve(name).replace("\n", "\n" + indent)

        def _resolve(name):
            if name == "tools":
                return self.tools_registry.signatures()
            if name == "allowed_paths":
                return _allowed_paths_slot()
            if name == "python_sandbox":
                return _python_sandbox_slot()
            if name == "python_sandbox_writes":
                return _python_sandbox_writes_slot()
            filler = self.env.slot(name)
            if filler is None:
                self.tools_registry.report_error(f"skill {skill_name!r}: no slot filler for {{{{{name}}}}}")
                return ""
            try:
                result = filler(SlotContext(agent=self.agent, skill=skill_name))
            except Exception as e:
                log.exception("skill %r: slot %r raised", skill_name, name)
                self.tools_registry.report_error(
                    f"skill {skill_name!r}: slot {name!r} raised: {type(e).__name__}: {e}")
                return f"[slot {name!r} failed]"
            if result is None:
                return ""
            return str(result)
        return _SLOT_RE.sub(fill, body)
