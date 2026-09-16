"""Tests for cai.usage: the token total read off an api usage report, shared by
the tui status line and the :agents attach view."""
from cai import usage


def test_total_tokens_prefers_total_then_sums_then_zero():
    assert usage.total_tokens({"total_tokens": 42, "prompt_tokens": 1}) == 42
    assert usage.total_tokens({"prompt_tokens": 10, "completion_tokens": 5}) == 15
    assert usage.total_tokens({}) == 0
    assert usage.total_tokens(None) == 0


def test_format_ctx_reads_unknown_until_a_sample():
    assert usage.format_ctx(0, 1000) == "ctx ? (?/1000)"
    assert usage.format_ctx(250, 1000) == "ctx 25% (250/1000)"


def test_agent_remembers_the_last_turns_tokens():
    from test_wired_agent import UsageApi, make_agent

    agent = make_agent(api=UsageApi(chunks=["a"], totals=[70, 90]))
    assert agent.tokens == 0
    agent.run("one").wait()
    assert agent.tokens == 70
    agent.run("two").wait()
    assert agent.tokens == 90
    assert agent.clone().tokens == 90
    assert agent.clone(overrides={"messages": []}).tokens == 0
