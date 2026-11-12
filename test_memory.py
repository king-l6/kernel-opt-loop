"""跨运行经验：按 (算子, 设备) 存最好版本、经验与失败，并用于 warm start。"""

import devices
import memory
from gates import Runner
from intake import load_task
from loop import Config
from scheduler import run_schedule

TASK = load_task("inbox/softmax")
SPEC = devices.detect("cpu")

STABLE = """
def softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)
"""


def _config(**overrides):
    config = Config.from_task(TASK)
    config.target_ms = None
    config.budget = 2
    config.repeats = 1
    config.warmup = 0
    config.threads = 1
    config.probe_budget = 0
    config.memory = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _trace(best="stable", source=STABLE, latency=0.3, warm_status=None):
    rounds = [
        {"round": 0, "name": "baseline", "status": "baseline", "latency_ms": 3.0, "hypothesis": "原版"},
        {
            "round": 1,
            "name": best,
            "status": "accepted",
            "latency_ms": latency,
            "hypothesis": "向量化",
            "speedup_vs_baseline": 3.0 / latency,
            "source": source,
        },
    ]
    if warm_status:
        rounds.insert(
            1,
            {
                "round": 1,
                "name": "warm-start",
                "status": warm_status,
                "latency_ms": None,
                "hypothesis": "上次最优",
                "message": "case 0: non-finite output",
                "error_kind": "non_finite",
            },
        )
    return {
        "task": "softmax",
        "best": best,
        "best_source": source,
        "rounds": rounds,
        "metrics": {"baseline_ms": 3.0, "best_ms": latency, "speedup": 3.0 / latency},
    }


def test_store_roundtrip_and_summary(tmp_path):
    store = memory.MemoryStore(tmp_path)
    store.update("softmax", SPEC, _trace())
    data = store.load("softmax", SPEC)
    assert data["runs"] == 1
    assert data["best"]["source"] == STABLE
    assert data["lessons"][0]["hypothesis"] == "向量化"
    assert store.path("softmax", SPEC).exists()

    view = memory.summary(data)
    assert view["best"]["latency_ms"] == 0.3
    assert "source" not in view["best"]


def test_better_previous_best_is_kept(tmp_path):
    store = memory.MemoryStore(tmp_path)
    store.update("softmax", SPEC, _trace(latency=0.2))
    data = store.update("softmax", SPEC, _trace(best="fresh", source=STABLE + "\n# v2\n", latency=0.4))
    assert data["runs"] == 2
    assert data["best"]["latency_ms"] == 0.2


def test_stale_best_is_dropped_after_warm_failure(tmp_path):
    store = memory.MemoryStore(tmp_path)
    store.update("softmax", SPEC, _trace(latency=0.2))
    trace = _trace(best="baseline", source=TASK.source, latency=3.0, warm_status="accuracy_fail")
    data = store.update("softmax", SPEC, trace)
    assert "best" not in data
    assert any(item["status"] == "accuracy_fail" for item in data["failures"])


class FakeStore:
    def __init__(self, data):
        self.data = data
        self.saved = None

    def load(self, task_name, spec):
        return self.data

    def update(self, task_name, spec, trace):
        self.saved = trace
        return self.data


class NoProposal:
    kind = "scripted"

    def __init__(self):
        self.ctx = None

    def propose(self, ctx, n=1):
        self.ctx = ctx
        return []


def test_warm_start_adopts_previous_source():
    store = FakeStore({"runs": 1, "best": {"source": STABLE, "hypothesis": "向量化", "latency_ms": 0.5}})
    config = _config()
    proposer = NoProposal()
    trace = run_schedule(TASK, proposer, Runner(TASK, config), config, memory_store=store)
    assert [item["name"] for item in trace["rounds"]] == ["baseline", "warm-start"]
    assert trace["rounds"][-1]["status"] == "accepted"
    assert trace["best"] == "warm-start"
    assert trace["experiments"] == 1
    assert store.saved is trace
    assert proposer.ctx["memory"]["best"]["latency_ms"] == 0.5


def test_warm_start_skips_reference_source():
    store = FakeStore({"runs": 1, "best": {"source": TASK.source, "latency_ms": 3.0}})
    config = _config()
    trace = run_schedule(TASK, NoProposal(), Runner(TASK, config), config, memory_store=store)
    assert [item["name"] for item in trace["rounds"]] == ["baseline"]
    assert trace["experiments"] == 0
