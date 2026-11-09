import math

from metrics import geomean, is_trace, summarize, trace_metrics


def _trace(name, baseline, best, target=None, experiments=3, tokens=0):
    return {
        "task": name,
        "device": "cpu",
        "proposer": "test",
        "elapsed_s": 1.0,
        "rounds": [{}, {}, {}],
        "baseline_ms": baseline,
        "best_latency_ms": best,
        "target_ms": target,
        "tokens": tokens,
        "metrics": {
            "baseline_ms": baseline,
            "best_ms": best,
            "speedup": baseline / best,
            "target_ms": target,
            "target_met": best <= target if target is not None else None,
            "experiments": experiments,
            "evaluations": experiments,
            "accepted": 1,
            "rounds": experiments,
            "tokens": tokens,
            "stop_reason": "budget",
        },
    }


def test_trace_metrics_without_target_uses_speedup():
    row = trace_metrics(_trace("a", 10.0, 2.0))
    assert row["success"] is True
    assert math.isclose(row["speedup"], 5.0)
    assert math.isclose(row["gain_per_experiment"], math.log(5.0) / 3)


def test_trace_metrics_with_target():
    row = trace_metrics(_trace("a", 10.0, 2.0, target=1.0))
    assert row["success"] is False
    assert row["target_met"] is False


def test_summarize_geomean_and_success_rate():
    traces = [
        _trace("a", 10.0, 2.0, target=3.0),
        _trace("b", 10.0, 5.0, target=4.0),
    ]
    summary = summarize(traces)
    assert summary["tasks"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert math.isclose(summary["geomean_speedup"], math.sqrt(5.0 * 2.0))


def test_geomean_ignores_missing_values():
    assert geomean([4.0, None, 0.0, 1.0]) == 2.0


def test_is_trace_rejects_old_handwritten_demo():
    assert is_trace(_trace("a", 10.0, 2.0)) is True
    assert is_trace({"kernel": "flash_attention_fwd", "rounds": [{}]}) is False
    assert is_trace({"task": "a", "rounds": "oops"}) is False
