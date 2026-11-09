"""把多条轨迹压成可比较的指标。

单个任务的“快了多少”不够回答 Agent 是否有效；这里统一算：
成功率、几何平均加速、轮数、单位实验收益、token 成本。
"""

from __future__ import annotations

import math
import statistics


def is_trace(data) -> bool:
    """只认本流水线产出的轨迹，旧的手写演示文件没有 task 字段。"""
    return (
        isinstance(data, dict)
        and bool(data.get("task"))
        and isinstance(data.get("rounds"), list)
    )


def trace_metrics(trace: dict) -> dict:
    metrics = trace.get("metrics") or {}
    baseline = metrics.get("baseline_ms", trace.get("baseline_ms"))
    best = metrics.get("best_ms", trace.get("best_latency_ms"))
    speedup = metrics.get("speedup", trace.get("speedup"))
    if speedup is None and baseline and best:
        speedup = baseline / best
    target = metrics.get("target_ms", trace.get("target_ms"))
    target_met = metrics.get("target_met")
    if target_met is None and target is not None and best is not None:
        target_met = best <= target
    experiments = int(metrics.get("experiments", 0) or 0)
    evaluations = int(metrics.get("evaluations", experiments) or 0)
    accepted = int(metrics.get("accepted", 0) or 0)
    elapsed = float(trace.get("elapsed_s") or 0.0)
    success = bool(target_met) if target is not None else bool(speedup and speedup >= 1.1)
    return {
        "task": trace.get("task"),
        "device": trace.get("device"),
        "proposer": trace.get("proposer"),
        "baseline_ms": baseline,
        "best_ms": best,
        "speedup": speedup,
        "target_ms": target,
        "target_met": bool(target_met) if target_met is not None else None,
        "success": success,
        "rounds": int(metrics.get("rounds", max(0, len(trace.get("rounds", [])) - 1)) or 0),
        "experiments": experiments,
        "evaluations": evaluations,
        "accepted": accepted,
        "accept_rate": (accepted / experiments) if experiments else None,
        "tokens": int(metrics.get("tokens", trace.get("tokens", 0)) or 0),
        "elapsed_s": elapsed,
        "gain_per_experiment": (math.log(speedup) / experiments) if speedup and experiments else None,
        "stop_reason": metrics.get("stop_reason", trace.get("stop_reason")),
    }


def summarize(traces: list[dict]) -> dict:
    rows = [trace_metrics(trace) for trace in traces]
    speedups = [row["speedup"] for row in rows if row["speedup"]]
    successes = [row for row in rows if row["success"]]
    rounds = [row["rounds"] for row in rows]
    tokens = sum(row["tokens"] for row in rows)
    experiments = sum(row["experiments"] for row in rows)
    elapsed = sum(row["elapsed_s"] for row in rows)
    return {
        "tasks": len(rows),
        "successes": len(successes),
        "success_rate": (len(successes) / len(rows)) if rows else None,
        "geomean_speedup": geomean(speedups),
        "mean_speedup": statistics.fmean(speedups) if speedups else None,
        "median_rounds": statistics.median(rounds) if rounds else None,
        "mean_rounds": statistics.fmean(rounds) if rounds else None,
        "accepted": sum(row["accepted"] for row in rows),
        "experiments": experiments,
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tokens_per_success": (tokens / len(successes)) if successes else None,
        "experiments_per_success": (experiments / len(successes)) if successes else None,
        "rows": rows,
    }


def geomean(values: list[float]) -> float | None:
    values = [value for value in values if value and value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def report_text(summary: dict) -> str:
    lines = [
        f"{'task':<16}{'rounds':>7}{'best':>12}{'speedup':>9}{'target':>8}{'exp':>5}{'tokens':>8}  stop",
        "-" * 80,
    ]
    for row in summary["rows"]:
        best = "-" if row["best_ms"] is None else f"{row['best_ms']:.3f}ms"
        speedup = "-" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        target = "yes" if row["target_met"] else ("no" if row["target_met"] is False else "-")
        lines.append(
            f"{str(row['task']):<16}{row['rounds']:>7}{best:>12}{speedup:>9}{target:>8}"
            f"{row['experiments']:>5}{row['tokens']:>8}  {row['stop_reason'] or ''}"
        )
    lines.append("-" * 80)
    geomean_text = "-" if summary["geomean_speedup"] is None else f"{summary['geomean_speedup']:.2f}x"
    rate = "-" if summary["success_rate"] is None else f"{summary['success_rate']:.0%}"
    lines.append(
        f"tasks={summary['tasks']}  success@target={rate}  geomean={geomean_text}  "
        f"experiments={summary['experiments']}  tokens={summary['tokens']}  "
        f"elapsed={summary['elapsed_s']:.1f}s"
    )
    if summary["experiments_per_success"] is not None:
        lines.append(
            f"per success: {summary['experiments_per_success']:.1f} experiments, "
            f"{summary['tokens_per_success']:.0f} tokens"
        )
    return "\n".join(lines)
