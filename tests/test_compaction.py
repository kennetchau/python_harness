"""Compaction + replay-consistency tests.

Regression tests for the mid-turn tool-chain compaction: the budget must be
checked inside the tool loop, a single long chain (no older user turns)
must still be compactable (round-level cut), and live memory must always
replay to itself after any number of compactions (including after a
session resume).
"""

import os
import tempfile

os.environ.setdefault("HARNESS_STATE", tempfile.mkdtemp())
os.environ.setdefault("HARNESS_WORKSPACE", tempfile.mkdtemp())

from src.harness import config as cfgmod
from src.harness.agent import Session
from src.harness.agent.compaction import (compact_once, ensure_budget,
                                          _find_round_cut)

SYS = "sys prompt"


def make_cfg(**ctx):
    raw = {"context": {"max_tokens": 8000, "summarize": "auto",
                       "summarize_threshold": 0.5, "keep_recent_turns": 2,
                       **ctx},
           "workspace": {"path": os.environ["HARNESS_WORKSPACE"]}}
    return cfgmod.build_config(raw)


class FakeClient:
    n = 0

    def complete(self, messages, max_tokens):
        FakeClient.n += 1
        return f"summary #{FakeClient.n}"


def assert_replay_consistent(session, live_mem):
    """live memory (sans system) must equal replay() (sans system)."""
    replayed = session.replay(SYS)
    live = [m for m in live_mem if m.get("role") != "system"]
    rp = [m for m in replayed if m.get("role") != "system"]
    assert live == rp, f"replay diverged: live={len(live)} msgs, replay={len(rp)}"


def assert_groups_intact(memory):
    """Every tool message must match a tool_call id of an earlier assistant."""
    ids = set()
    for m in memory:
        if m.get("role") == "assistant":
            ids.update(tc["id"] for tc in m.get("tool_calls") or [])
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in ids, "dangling tool result after compaction"


def round(i, size=1000):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "run_command", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "z" * size},
    ]


def test_multi_turn_compaction_and_resume():
    cfg = make_cfg()
    session = Session.create("coder", "m")
    mem = [{"role": "system", "content": SYS}]
    for i in range(10):
        session.record_user(f"task {i}")
        mem.append({"role": "user", "content": f"task {i} " + "x" * 800})
        session.record_assistant(f"done {i}")
        mem.append({"role": "assistant", "content": f"done {i} " + "y" * 800})
    mem = ensure_budget(mem, FakeClient(), cfg, session) or mem
    assert_replay_consistent(session, mem)
    assert_replay_consistent(Session.open(session.path), mem)
    assert_groups_intact(mem)
    # second compaction: two summaries, still consistent
    for i in range(10, 20):
        session.record_user(f"task {i}")
        mem.append({"role": "user", "content": f"task {i} " + "x" * 800})
        session.record_assistant(f"done {i}")
        mem.append({"role": "assistant", "content": f"done {i} " + "y" * 800})
    mem = ensure_budget(mem, FakeClient(), cfg, session) or mem
    assert_replay_consistent(session, mem)
    assert_replay_consistent(Session.open(session.path), mem)


def test_single_long_tool_chain_compacts_mid_turn():
    """One user turn + a long tool chain: nothing 'old', but compaction must
    still fire via a round-level cut (regression: used to return None and,
    with a leading summary present, loop forever on an empty span)."""
    cfg = make_cfg()
    session = Session.create("coder", "m")
    session.record_user("one big task")
    mem = [{"role": "system", "content": SYS},
           {"role": "user", "content": "one big task"}]
    for i in range(30):
        mem += round(i)
    cut = _find_round_cut(mem, 1, 2)
    assert cut is not None and cut > 1, "round cut must drop real rounds"
    before = len(mem)
    mem = compact_once(mem, FakeClient(), cfg, session)
    assert mem is not None and len(mem) < before, "compaction must shrink memory"
    assert_replay_consistent(session, mem)
    assert_replay_consistent(Session.open(session.path), mem)
    assert_groups_intact(mem)
    # a further round cut: summaries accumulate, consistency holds
    for i in range(30, 60):
        mem += round(i, size=1000)
    mem = compact_once(mem, FakeClient(), cfg, session)
    assert mem is not None
    assert_replay_consistent(session, mem)
    assert_groups_intact(mem)


def test_round_cut_keeps_trailing_answer_and_inflight_round():
    session = Session.create("coder", "m")
    session.record_user("task")
    mem = [{"role": "system", "content": SYS}, {"role": "user", "content": "task"}]
    for i in range(10):
        mem += round(i, size=1500)
    # in-flight round: 2 calls, only 1 result
    mem.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "g0", "type": "function", "function": {"name": "x", "arguments": "{}"}},
        {"id": "g1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]})
    mem.append({"role": "tool", "tool_call_id": "g0", "content": "r"})
    mem2 = compact_once(mem, FakeClient(), cfgmod.build_config({
        "context": {"max_tokens": 8000, "summarize": "auto",
                    "summarize_threshold": 0.5, "keep_recent_turns": 2},
        "workspace": {"path": os.environ["HARNESS_WORKSPACE"]}}), session)
    assert mem2 is not None
    assert mem2[-2]["role"] == "assistant" and len(mem2[-2]["tool_calls"]) == 2
    assert mem2[-1]["role"] == "tool" and mem2[-1]["tool_call_id"] == "g0"
    assert_replay_consistent(session, mem2)
    assert_groups_intact(mem2)


def test_ensure_budget_makes_progress_or_bails():
    """If a pass ever fails to shrink memory, ensure_budget must bail
    (regression: empty-span 'compaction' looped forever when a leading
    summary existed and the cut collapsed onto base)."""
    import src.harness.agent.compaction as comp
    cfg = make_cfg()
    session = Session.create("coder", "m")
    mem = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "[Summary of prior conversation]\nold"},
        {"role": "user", "content": "new " + "x" * 20000},
        {"role": "assistant", "content": "a" * 20000},
    ]
    orig = comp._find_cut
    comp._find_cut = lambda m, k: 2   # force cut == base (empty span)
    try:
        out = comp.ensure_budget(mem, FakeClient(), cfg, session)
    finally:
        comp._find_cut = orig
    assert out is None, "ensure_budget must abort when compaction makes no progress"


def test_loop_checks_budget_mid_chain():
    """End-to-end through run_turn: a single turn whose tool rounds each add
    ~20k chars must trigger compaction INSIDE the tool chain, not only at
    the next turn start."""
    from src.harness.agent import AgentLoop
    from src.harness.backend.client import (TextDelta, ToolCallDelta, Done,
                                            Usage)

    cfg = cfgmod.build_config({
        "context": {"max_tokens": 12000, "summarize": "auto",
                    "summarize_threshold": 0.7, "keep_recent_turns": 2},
        "workspace": {"path": os.environ["HARNESS_WORKSPACE"], "auto_git": False},
        "tools": {"limits": {"exec_output_chars": 100000,
                             "max_tool_rounds": 50, "exec_timeout_sec": 10}},
    })

    class ChainClient:
        def __init__(self):
            self.round = 0
            self.compact_calls = 0
            self.prompt_sizes = []

        def stream_chat(self, messages, tools=None, cancel=None):
            import json as _json
            self.round += 1
            self.prompt_sizes.append(
                sum(len(m.get("content") or "") for m in messages) // 4)
            if self.round <= 6:
                args = _json.dumps({"command": "head -c 20000 /dev/zero | tr '\\0' z"})
                yield ToolCallDelta(0, f"call_{self.round}", "run_command", args)
                yield Done("tool_calls", Usage(prompt_tokens=1, completion_tokens=10))
            else:
                yield TextDelta("done")
                yield Done("stop", Usage(prompt_tokens=1, completion_tokens=5))

        def complete(self, messages, max_tokens):
            self.compact_calls += 1
            return "compact summary of earlier rounds"

    session = Session.create("coder", "m")
    client = ChainClient()
    loop = AgentLoop(cfg, client, session,
                     ask_approval=lambda n, a, p: "y",
                     emit=lambda k, d: None)
    res = loop.run_turn("do a long multi-step task")
    assert res.ok, f"turn failed: {res.reason}"
    assert client.compact_calls >= 1, "compaction must fire inside the tool chain"
    # unsummarized the 6 rounds would be ~33k est tokens; it must stay small
    assert max(client.prompt_sizes) < 30000
    # session log still replays to the live memory
    assert_replay_consistent(Session.open(session.path), loop.memory)
