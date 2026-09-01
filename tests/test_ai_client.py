"""Cache, retry and parsing behind one call."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinels.backtest.agents.client import DecisionClient
from sentinels.backtest.storage.cache import AICache, CacheMiss
from tests.conftest import ScriptedLLM, decision_reply

TS = datetime(2026, 1, 15, tzinfo=timezone.utc)
PAYLOAD = {"account": {"equity": 1000.0}, "market": {"BTCUSDT": {"close": 42000.0}}}
REPLY = decision_reply({"symbol": "BTCUSDT", "action": "wait"})


def client(llm, cache=None, **kwargs):
    kwargs.setdefault("variant", "basic")
    kwargs.setdefault("retry_base_delay", 0.0)
    return DecisionClient(llm, cache or AICache(None), **kwargs)


def test_a_decision_is_fetched_once_and_then_served_from_cache():
    llm = ScriptedLLM([REPLY])
    c = client(llm)

    first = c.decide(PAYLOAD, "sys", "user", TS)
    second = c.decide(PAYLOAD, "sys", "user", TS)

    assert first.from_cache is False
    assert second.from_cache is True
    assert len(llm.calls) == 1, "identical state must not be paid for twice"
    assert c.calls == 1


def test_different_state_produces_a_different_key():
    c = client(ScriptedLLM([REPLY, REPLY]))
    a = c.decide(PAYLOAD, "sys", "user", TS)
    b = c.decide({**PAYLOAD, "account": {"equity": 900.0}}, "sys", "user", TS)
    assert a.cache_key != b.cache_key


def test_the_variant_is_part_of_the_key():
    cache = AICache(None)
    basic = client(ScriptedLLM([REPLY]), cache, variant="basic")
    aggressive = client(ScriptedLLM([REPLY]), cache, variant="aggressive")

    assert (
        basic.decide(PAYLOAD, "s", "u", TS).cache_key
        != aggressive.decide(PAYLOAD, "s", "u", TS).cache_key
    )


def test_prompts_are_part_of_the_key():
    c = client(ScriptedLLM([REPLY, REPLY, REPLY]))
    base = c.decide(PAYLOAD, "system with memo A", "user", TS)
    changed_system = c.decide(PAYLOAD, "system with memo B", "user", TS)
    changed_user = c.decide(PAYLOAD, "system with memo A", "different user prompt", TS)

    assert len({base.cache_key, changed_system.cache_key, changed_user.cache_key}) == 3


def test_model_identity_is_part_of_the_key():
    cache = AICache(None)
    first_llm = ScriptedLLM([REPLY])
    second_llm = ScriptedLLM([REPLY])
    first_llm.model = "model-a"
    second_llm.model = "model-b"

    first = client(first_llm, cache).decide(PAYLOAD, "s", "u", TS)
    second = client(second_llm, cache).decide(PAYLOAD, "s", "u", TS)
    assert first.cache_key != second.cache_key


def test_replay_only_raises_instead_of_calling_the_provider():
    llm = ScriptedLLM([REPLY])
    c = client(llm, replay_only=True)

    with pytest.raises(CacheMiss):
        c.decide(PAYLOAD, "sys", "user", TS)
    assert llm.calls == []


def test_replay_only_is_satisfied_by_a_warm_cache():
    cache = AICache(None)
    client(ScriptedLLM([REPLY]), cache).decide(PAYLOAD, "sys", "user", TS)

    offline = client(ScriptedLLM([]), cache, replay_only=True)
    result = offline.decide(PAYLOAD, "sys", "user", TS)

    assert result.from_cache is True
    assert offline.calls == 0


def test_a_transient_failure_is_retried():
    llm = ScriptedLLM([REPLY], fail_times=2)
    result = client(llm).decide(PAYLOAD, "sys", "user", TS)

    assert result.attempts == 3
    assert result.error == ""
    assert result.parsed.decisions[0].action == "wait"


def test_exhausting_the_retries_yields_a_wait_rather_than_an_exception():
    llm = ScriptedLLM([], fail_times=99)
    result = client(llm, max_retries=2).decide(PAYLOAD, "sys", "user", TS)

    assert "model call failed after 2 attempts" in result.error
    assert result.parsed.fallback is True
    assert result.parsed.decisions[0].action == "wait"


def test_a_failed_call_is_not_cached():
    cache = AICache(None)
    c = client(ScriptedLLM([], fail_times=99), cache, max_retries=1)
    c.decide(PAYLOAD, "sys", "user", TS)
    assert cache.entries == {}, "caching a failure would poison every later replay"


def test_disabling_the_cache_calls_the_provider_every_time():
    llm = ScriptedLLM([REPLY, REPLY])
    c = client(llm, cache_ai=False)
    c.decide(PAYLOAD, "sys", "user", TS)
    c.decide(PAYLOAD, "sys", "user", TS)
    assert len(llm.calls) == 2
