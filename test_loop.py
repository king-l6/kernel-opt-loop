import json

import pytest

from intake import load_task
from loop import Config, replay, run
from metrics import summarize


def _config():
    task = load_task("inbox/softmax")
    config = Config.from_task(task)
    config.target_ms = None
    config.repeats = 3
    config.warmup = 1
    config.threads = 1
    return config


def test_scripted_softmax_closed_loop(tmp_path):
    out = tmp_path / "trace.json"
    trace = run("inbox/softmax", config=_config(), proposer_kind="scripted", out_path=out)
    statuses = {item["name"]: item["status"] for item in trace["rounds"]}
    assert statuses["broken_syntax"] == "compile_error"
    assert statuses["wrong_axis"] == "accuracy_fail"
    assert statuses["unstable"] == "accuracy_fail"
    assert statuses["stable_vec"] == "accepted"
    assert statuses["redundant"] == "rejected_slower"
    assert trace["best"] == "stable_vec"
    assert trace["metrics"]["speedup"] > 1.5
    for item in trace["rounds"]:
        assert isinstance(item["feedback_out"], dict)
        assert item["steps"]
        if item["status"] in {"compile_error", "accuracy_fail"}:
            assert item["latency_ms"] is None
    assert out.exists()


def test_replay_reproduces_the_gates(tmp_path):
    out = tmp_path / "trace.json"
    run("inbox/softmax", config=_config(), proposer_kind="scripted", out_path=out)
    report = replay(out)
    assert report["ok"]
    assert report["checked"] >= 5


def test_replay_rejects_tampered_status(tmp_path):
    out = tmp_path / "trace.json"
    run("inbox/softmax", config=_config(), proposer_kind="scripted", out_path=out)
    trace = json.loads(out.read_text(encoding="utf-8"))
    for item in trace["rounds"]:
        if item["name"] == "broken_syntax":
            item["status"] = "accepted"
    out.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(RuntimeError, match="replay mismatch"):
        replay(out)


def test_summarize_reports_success_and_geomean(tmp_path):
    out = tmp_path / "trace.json"
    trace = run("inbox/softmax", config=_config(), proposer_kind="scripted", out_path=out)
    summary = summarize([trace])
    assert summary["tasks"] == 1
    assert summary["successes"] == 1
    assert summary["geomean_speedup"] > 1.5
    assert summary["experiments"] >= 4
    assert summary["rows"][0]["task"] == "softmax"
