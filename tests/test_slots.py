"""Tests for skill-body slots: {{name}} holes in a skill body filled fresh on
every system_prompt read by cai.slot fillers ({{tools}} stays the builtin).
Fully offline: skills resolve from a synthetic extension dir, no LLM, no MCP."""
import os
import textwrap

from cai.environment import Environment, Extension
from cai.skills import SkillsRegistry, SlotContext, slot
from cai.tools import ToolsRegistry


def _env_with_skill(tmp_path, name, text):
    """an Environment whose one extension carries skills/<name>.md."""
    skills_dir = tmp_path / "ext" / "skills"
    os.makedirs(skills_dir, exist_ok=True)
    with open(skills_dir / (name + ".md"), "w") as f:
        f.write(textwrap.dedent(text))
    return Environment([Extension(name="ext", path=str(tmp_path / "ext"))])


def _registry(env, skill, agent=None):
    tools_registry = ToolsRegistry(env)
    return SkillsRegistry.for_skills([skill],
                                     tools_registry=tools_registry,
                                     env=env,
                                     agent=agent)


def test_slot_fills_skill_body(tmp_path):
    env = _env_with_skill(tmp_path, "memo", """\
        name: memo
        ---
        Notes:
        {{notes}}
        """)

    def notes(ctx):
        return "first\nsecond"
    env.register_slot(notes)

    registry = _registry(env, "memo")
    assert "Notes:\nfirst\nsecond" in registry.system_prompt


def test_slot_resolves_fresh_on_every_read(tmp_path):
    env = _env_with_skill(tmp_path, "memo", "body {{counter}}")
    calls = []

    def counter(ctx):
        calls.append(1)
        return str(len(calls))
    env.register_slot(counter)

    registry = _registry(env, "memo")
    assert "body 1" in registry.system_prompt
    assert "body 2" in registry.system_prompt


def test_slot_context_carries_agent_and_skill(tmp_path):
    env = _env_with_skill(tmp_path, "memo", "{{probe}}")
    seen = []

    def probe(ctx):
        seen.append(ctx)
        return "ok"
    env.register_slot(probe)

    marker = object()
    registry = _registry(env, "memo", agent=marker)
    registry.system_prompt
    assert len(seen) == 1
    assert isinstance(seen[0], SlotContext)
    assert seen[0].agent is marker
    assert seen[0].skill == "memo"


def test_unknown_slot_fills_empty(tmp_path):
    env = _env_with_skill(tmp_path, "memo", "before {{missing}} after")
    registry = _registry(env, "memo")
    assert registry.system_prompt == "before  after"


def test_raising_slot_fills_error_marker(tmp_path):
    env = _env_with_skill(tmp_path, "memo", "state: {{broken}}")

    def broken(ctx):
        raise RuntimeError("boom")
    env.register_slot(broken)

    registry = _registry(env, "memo")
    assert "state: [slot 'broken' failed]" in registry.system_prompt


def test_none_result_fills_empty(tmp_path):
    env = _env_with_skill(tmp_path, "memo", "x{{quiet}}y")

    def quiet(ctx):
        return None
    env.register_slot(quiet)

    registry = _registry(env, "memo")
    assert registry.system_prompt == "xy"


def test_tools_slot_still_builtin(tmp_path):
    env = _env_with_skill(tmp_path, "memo", "tools:\n{{tools}}")

    def echo(text: str) -> str:
        """Echo text back."""
        return text
    env.register_tool(echo)

    tools_registry = ToolsRegistry(env)
    tools_registry.select("echo")
    registry = SkillsRegistry.for_skills(["memo"], tools_registry=tools_registry, env=env)
    assert "echo" in registry.system_prompt
    assert "{{tools}}" not in registry.system_prompt


def test_slot_decorator_registers_on_target_env():
    env = Environment()
    Environment._default = env

    @slot
    def notes(ctx):
        return "n"

    assert env.slot("notes") is notes
    assert notes._cai_slot_name == "notes"


def test_slot_registered_during_extension_load_is_namespaced():
    env = Environment()
    env._loading_extension = "memory"

    def notes(ctx):
        return "n"
    env.register_slot(notes)

    assert env.slot("notes") is None
    assert env.slot("memory__notes") is notes
    assert notes._cai_slot_name == "memory__notes"


def test_agent_wires_itself_into_slot_context(tmp_path):
    import cai.agent as agent_module
    env = _env_with_skill(tmp_path, "memo", "{{whoami}}")
    seen = []

    def whoami(ctx):
        seen.append(ctx.agent)
        return "me"
    env.register_slot(whoami)

    agent = agent_module.Agent(model="m", api=object(), env=env, skills=["memo"])
    try:
        assert "me" in agent.system_prompt
        assert seen[0] is agent
    finally:
        agent.close()
