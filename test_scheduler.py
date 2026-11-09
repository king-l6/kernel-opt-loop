from intake import load_task
from loop import Config
from proposers import Patch
from scheduler import run_schedule

TASK = load_task("inbox/softmax")


class FakeResult:
    def __init__(self, status, latency=None, message="", error_kind=""):
        self.status = status
        self.latency_ms = latency
        self.message = message or status
        self.max_abs_err = None
        self.max_rel_err = None
        self.profile = None
        self.stats = {"p90_ms": latency, "cv": 0.01} if latency else None
        self.error_kind = error_kind
        self.wall_s = 0.0


class FakeRunner:
    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = []

    def check(self, name, source, profile=False):
        self.calls.append(name)
        if not self.plan:
            return FakeResult("ok", 8.0)
        return self.plan.pop(0)

    def profile(self, name, source):
        return [{"name": "aten::exp", "us": 10.0, "pct": 50.0}]


class QueueProposer:
    kind = "fake"

    def __init__(self, batches=None):
        self.batches = list(batches or [])
        self.contexts = []

    def propose(self, ctx, n=1):
        self.contexts.append(ctx)
        return self.batches.pop(0) if self.batches else []


class EndlessProposer:
    kind = "fake"

    def __init__(self):
        self.index = 0

    def propose(self, ctx, n=1):
        self.index += 1
        source = f"def softmax(x):\n    return x + {self.index}\n"
        return [Patch(name=f"p{self.index}", title=f"p{self.index}", source=source, hypothesis="试一版")]


def _config(**overrides):
    values = dict(repeats=1, warmup=0, threads=1, device="cpu", target_ms=None)
    values.update(overrides)
    return Config(**values)


def _patch(name, source, title=None):
    return Patch(name=name, title=title or name, source=source, hypothesis="试一版")


def test_timeout_then_next_candidate_is_accepted():
    proposer = QueueProposer(
        [
            [_patch("timeout", "def softmax(x):\n    return x + 1\n")],
            [_patch("fast", "def softmax(x):\n    return x * 1\n")],
        ]
    )
    runner = FakeRunner(
        [
            FakeResult("ok", 10.0),
            FakeResult("timeout", None, "killed after 2s"),
            FakeResult("ok", 4.0),
        ]
    )
    trace = run_schedule(TASK, proposer, runner, _config(budget=4))
    assert [item["status"] for item in trace["rounds"]] == ["baseline", "timeout", "accepted"]
    assert trace["best"] == "fast"
    assert proposer.contexts[0]["feedback"]["status"] == "baseline"
    assert proposer.contexts[1]["feedback"]["status"] == "timeout"


def test_patience_stops_after_repeated_non_improvements():
    runner = FakeRunner([FakeResult("ok", 10.0)] + [FakeResult("ok", 10.5)] * 5)
    trace = run_schedule(TASK, EndlessProposer(), runner, _config(budget=10, patience=2))
    assert trace["stop_reason"] == "patience"
    assert [item["status"] for item in trace["rounds"]] == [
        "baseline",
        "rejected_slower",
        "rejected_slower",
    ]


def test_duplicate_source_does_not_hit_the_sandbox_twice():
    source = "def softmax(x):\n    return x + 7\n"
    proposer = QueueProposer(
        [[_patch("first", source)], [_patch("again", source)]]
    )
    runner = FakeRunner([FakeResult("ok", 10.0), FakeResult("ok", 4.0)])
    trace = run_schedule(TASK, proposer, runner, _config(budget=4))
    assert [item["status"] for item in trace["rounds"]] == ["baseline", "accepted", "duplicate"]
    assert runner.calls == ["baseline", "first"]


def test_policy_error_is_rejected_before_sandbox():
    proposer = QueueProposer(
        [[_patch("policy", "import os\ndef softmax(x):\n    return x\n")]]
    )
    runner = FakeRunner([FakeResult("ok", 10.0)])
    trace = run_schedule(TASK, proposer, runner, _config(budget=4))
    assert trace["rounds"][1]["status"] == "policy_error"
    assert runner.calls == ["baseline"]


def test_feedback_is_structured_not_a_bare_string():
    proposer = QueueProposer([])
    runner = FakeRunner([FakeResult("ok", 10.0)])
    run_schedule(TASK, proposer, runner, _config(budget=1))
    ctx = proposer.contexts[0]
    assert isinstance(ctx["feedback"], dict)
    assert "text" in ctx["feedback"]
    assert "def softmax" in ctx["current_best"]["source"]
