"""Agent 内循环：试跑工具、预算、以及和 scheduler 的接线。"""

import json

from gates import Runner
from intake import load_task
from loop import Config
from probes import ProbeSession
from proposers import AgentProposer, Completion, Patch
from scheduler import run_schedule

TASK = load_task("inbox/softmax")

STABLE = """
def softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)
"""


class FakeClient:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, messages, max_tokens=4096, temperature=0.7):
        self.calls.append(list(messages))
        return Completion(self.texts.pop(0), {"total_tokens": 10, "input_tokens": 6, "output_tokens": 4})


def _ctx(probe=None):
    return {
        "task_spec": {"name": "softmax", "entry": "softmax"},
        "reference_source": STABLE,
        "current_best": {"name": "baseline", "source": STABLE, "latency_ms": 10.0},
        "baseline_ms": 10.0,
        "target_ms": 1.0,
        "feedback": {"status": "baseline", "text": "基线耗时 10.000 ms。", "profile": []},
        "history": [],
        "round": 1,
        "budget_left": 8,
        "best_of": 1,
        "device": "cpu",
        "triton_ready": False,
        "probe": probe,
        "probe_budget": 3,
    }


def _probe_ok(calls, latency=1.0):
    def probe(source, profile=False, name="probe"):
        calls.append({"source": source, "profile": profile, "name": name})
        return {
            "status": "ok",
            "message": "accuracy passed",
            "latency_ms": latency,
            "probe_left": 2,
            "profile": [{"name": "aten::exp", "us": 5.0, "pct": 50.0}] if profile else [],
        }

    return probe


def _config(**overrides):
    config = Config.from_task(TASK)
    config.target_ms = None
    config.budget = 1
    config.repeats = 1
    config.warmup = 0
    config.threads = 1
    config.probe_budget = 2
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_agent_probes_then_submits():
    calls = []
    client = FakeClient(
        [
            json.dumps({"action": "probe", "source": STABLE, "profile": True}),
            json.dumps({"action": "submit", "source": STABLE, "hypothesis": "向量化"}),
        ]
    )
    proposer = AgentProposer(client, "softmax", max_steps=4)
    patches = proposer.propose(_ctx(_probe_ok(calls)), n=1)
    assert len(patches) == 1
    assert calls == [{"source": STABLE, "profile": True, "name": "agent-r1"}]
    assert patches[0].usage["total_tokens"] == 20
    assert patches[0].meta == {"steps": 2, "probes": 1}
    assert len(client.calls) == 2


def test_agent_repairs_bad_json_then_submits():
    client = FakeClient(
        [
            "这不是 JSON",
            json.dumps({"action": "submit", "source": STABLE, "hypothesis": "ok"}),
        ]
    )
    patches = AgentProposer(client, "softmax", max_steps=3).propose(_ctx(_probe_ok([])), n=1)
    assert len(patches) == 1
    assert len(client.calls) == 2


def test_agent_gives_up():
    client = FakeClient([json.dumps({"action": "give_up", "hypothesis": "没有空间"})])
    proposer = AgentProposer(client, "softmax", max_steps=3)
    assert proposer.propose(_ctx(_probe_ok([])), n=1) == []
    assert "没有空间" in proposer.last_error


def test_agent_returns_best_probe_when_steps_run_out():
    client = FakeClient([json.dumps({"action": "probe", "source": STABLE, "profile": False})])
    proposer = AgentProposer(client, "softmax", max_steps=1)
    patches = proposer.propose(_ctx(_probe_ok([], latency=0.5)), n=1)
    assert len(patches) == 1
    assert patches[0].source == STABLE
    assert patches[0].meta["exhausted"] is True
    assert patches[0].meta["probes"] == 1


def test_probe_cache_and_budget():
    config = _config(probe_budget=1)
    session = ProbeSession(TASK, Runner(TASK, config), config)
    first = session.probe(STABLE)
    assert first["status"] == "ok"
    assert session.used == 1
    cached = session.probe(STABLE)
    assert cached["cached"] is True
    assert session.used == 1
    over = session.probe("def softmax(x):\n    return x * 2\n")
    assert over["status"] == "probe_budget"


def test_probe_rejects_policy_violations_without_spending_budget():
    config = _config(probe_budget=1)
    session = ProbeSession(TASK, Runner(TASK, config), config)
    result = session.probe("def softmax(x):\n    torch.allclose = lambda *a, **k: True\n    return x\n")
    assert result["status"] == "policy_error"
    assert session.used == 0
    assert session.left == 1


def test_scheduler_exposes_probe_without_consuming_experiment_budget():
    class ToolProposer:
        kind = "tool"

        def propose(self, ctx, n=1):
            assert ctx["probe_budget"] == 2
            result = ctx["probe"](STABLE, profile=False, name="tool")
            assert result["status"] == "ok"
            return [Patch(name="tool", title="tool", source=STABLE, hypothesis="试跑后提交")]

    config = _config()
    trace = run_schedule(TASK, ToolProposer(), Runner(TASK, config), config)
    assert trace["experiments"] == 1
    assert trace["probes"] == 1
    assert trace["metrics"]["probes"] == 1
    assert trace["rounds"][-1]["status"] == "accepted"
    assert len(trace["rounds"]) == 2


def test_probe_only_agent_does_not_spend_experiments():
    class ProbeOnly:
        kind = "tool"

        def propose(self, ctx, n=1):
            ctx["probe"](STABLE, profile=False, name="tool")
            return []

    config = _config()
    trace = run_schedule(TASK, ProbeOnly(), Runner(TASK, config), config)
    assert trace["experiments"] == 0
    assert trace["probes"] == 1
    assert trace["stop_reason"] == "no_proposal"
