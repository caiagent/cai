name: subagents
tools: launch_agent, wait_agent, list_agents, kill_agent
---
# Skill: Sub-Agents

Delegate self-contained or parallelizable subtasks to background sub-agents, then collect each result by name. Don't delegate trivial single-tool steps — do those yourself.

The tools (pass arguments as named fields):
- `launch_agent(prompt="...", name="audit-auth-flow", skills=["fs-read-only"])` — start ONE sub-agent, returns its `agent_id` (the name) and what it was granted. Pass `tools=[...]` and/or `skills=[...]` — at least one (`tools=[]` for none). Optional: `model="..."`, `system_prompt="..."`.
- `wait_agent(agent_id="audit-auth-flow")` — block for ONE sub-agent's answer. A finished agent keeps its answer parked, so a late call still returns it.
- `list_agents()` — every sub-agent's status (running / finished / failed), without blocking.
- `kill_agent(agent_id="audit-auth-flow")` — stop ONE runaway sub-agent.

`agent_id` is a single string — the exact name `launch_agent` returned — ONE per call, never a list. To wait on several, make several `wait_agent` calls.

Rules:
- Name every agent descriptively, lowercase-dashed: `audit-auth-flow`, `summarize-test-logs`.
- Prompts must be self-contained — the sub-agent shares your working directory but sees nothing of your conversation. Include paths, constraints, and the expected output format.
- Grant only what the task needs via `tools`/`skills` (each a list, only a subset of your own). The child gets exactly what you list (a skill brings its own tools) — nothing is inherited. Check the launch result for dropped names.
- `skills` takes the `skill:` name shown above each skill in your prompt (`fs-read-only`), never the `# Skill:` title.
- Collect every result before your final answer. If one fails, retry with a sharper prompt or do it yourself.
