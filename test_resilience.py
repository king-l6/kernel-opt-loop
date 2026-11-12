"""生成器临时故障的容错：网关 5xx 不应该终止整个实验。"""

from intake import load_task
from loop import Config
from proposers import LLMError, Patch
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

    def check(self, name, source, profile=False):
        if not self.plan:
            return FakeResult("ok", 8.0)
        return self.plan.pop(0)

    def profile(self, name, source):
        return [{"name": "aten::exp", "us": 10.0, "pct": 50.0}]


class FlakyProposer:
    kind = "llm"

    def __init__(self, kinds):
        self.kinds = list(kinds)
        self.calls = 0
        self.last_error = ""
        self.last_error_kind = ""

    def propose(self, ctx, n=1):
        self.calls += 1
        kind = self.kinds.pop(0) if self.kinds else ""
        if kind in {"transient", "parse", "declined"}:
            self.last_error = f"模拟 {kind} 故障"
            self.last_error_kind = kind
            return []
        return [
            Patch(
                name=f"p{self.calls}",
                title="固定候选",
                source=f"def softmax(x):\n    return x + {self.calls}\n",
                hypothesis="测试用",
            )
        ]


def _config(**overrides):
    values = dict(repeats=1, warmup=0, threads=1, device="cpu", target_ms=None)
    values.update(overrides)
    return Config(**values)


def test_llm_error_carries_kind():
    assert LLMError("boom", "transient").kind == "transient"
    assert LLMError("boom").kind == "fatal"


def test_transient_failure_is_retried_without_stopping():
    proposer = FlakyProposer(["transient", "ok"])
    runner = FakeRunner([FakeResult("ok", 10.0), FakeResult("ok", 4.0)])
    trace = run_schedule(TASK, proposer, runner, _config(budget=1))
    assert proposer.calls == 2
    assert [item["status"] for item in trace["rounds"]] == ["baseline", "accepted"]
    assert trace["stop_reason"] == "budget"


def test_two_transient_failures_in_a_row_are_tolerated():
    proposer = FlakyProposer(["transient", "transient", "ok"])
    runner = FakeRunner([FakeResult("ok", 10.0), FakeResult("ok", 4.0)])
    trace = run_schedule(TASK, proposer, runner, _config(budget=1))
    assert proposer.calls == 3
    assert [item["status"] for item in trace["rounds"]] == ["baseline", "accepted"]


def test_repeated_transient_failures_end_as_proposer_error():
    proposer = FlakyProposer(["transient"] * 3)
    runner = FakeRunner([FakeResult("ok", 10.0)])
    trace = run_schedule(TASK, proposer, runner, _config(budget=4))
    assert proposer.calls == 3
    assert trace["stop_reason"] == "proposer_error"
    assert trace["experiments"] == 0


def test_declined_proposal_stops_immediately_as_no_proposal():
    proposer = FlakyProposer(["declined"])
    runner = FakeRunner([FakeResult("ok", 10.0)])
    trace = run_schedule(TASK, proposer, runner, _config(budget=4))
    assert proposer.calls == 1
    assert trace["stop_reason"] == "no_proposal"
