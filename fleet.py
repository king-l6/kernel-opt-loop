"""把一批算子铺到多块设备上跑，并按 (算子, 设备) 汇总。

调度规则只有一条：**同一块设备上的任务串行，不同设备之间并行**。
同一张卡上并发跑两个测速任务会互相抢带宽和频率，数字不可比；多卡之间则
可以并行，因为每块卡有自己的内存和时钟。

显式写了但不可用的设备直接跳过并记录原因，不会悄悄退回 CPU——否则
“在 A100 上跑过”这类结论会被 CPU 数字污染。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import devices as device_mod
from gates import best_device
from intake import list_task_dirs, load_task
from loop import Config, run
from metrics import geomean


def run_fleet(
    task_dirs,
    device_list,
    *,
    proposer: str = "auto",
    budget: int | None = None,
    out_dir: str | Path = "traces/fleet",
    jobs: int | None = None,
) -> dict:
    tasks = [Path(item) for item in task_dirs]
    device_list = list(dict.fromkeys(device_list))  # 去重且保序
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    def run_device(device: str) -> list[dict]:
        if not device_mod.available(device):
            return [
                _row(task, device, "skipped", error=f"设备不可用：{device}")
                for task in tasks
            ]
        rows = []
        for task_dir in tasks:
            rows.append(_run_one(task_dir, device, output, proposer, budget))
        return rows

    rows: list[dict] = []
    workers = max(1, min(jobs or len(device_list), len(device_list)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(run_device, device_list):
            rows.extend(chunk)

    summary = summarize(rows, tasks, device_list)
    (output / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize(rows: list[dict], tasks: list[Path], device_list: list[str]) -> dict:
    ok = [row for row in rows if row["status"] == "ok"]
    speedups = [row["speedup"] for row in ok if row.get("speedup")]
    per_device = []
    for device in device_list:
        spec = device_mod.detect(device)
        own = [row for row in rows if row["device"] == device]
        per_device.append(
            {
                "device": device,
                "name": spec.name,
                "peak_bw_gbps": spec.peak_bw_gbps,
                "available": device_mod.available(device),
                "runs": len(own),
                "successes": sum(1 for row in own if row.get("target_met")),
            }
        )
    return {
        "tasks": [Path(task).name for task in tasks],
        "devices": per_device,
        "rows": rows,
        "totals": {
            "runs": len(rows),
            "ok": len(ok),
            "failed": sum(1 for row in rows if row["status"] == "error"),
            "skipped": sum(1 for row in rows if row["status"] == "skipped"),
            "successes": sum(1 for row in ok if row.get("target_met")),
            "geomean_speedup": geomean(speedups),
            "tokens": sum(row.get("tokens") or 0 for row in ok),
            "experiments": sum(row.get("experiments") or 0 for row in ok),
            "probes": sum(row.get("probes") or 0 for row in ok),
            "elapsed_s": round(sum(row.get("elapsed_s") or 0 for row in rows), 2),
        },
    }


def matrix_text(summary: dict) -> str:
    """每个算子一行，每块设备一列，格子里是加速比和是否达标。"""
    columns = []
    for device in summary["devices"]:
        if not device.get("available"):
            columns.append(f"{device['device']}:不可用")
            continue
        peak = "" if device["peak_bw_gbps"] is None else f" {device['peak_bw_gbps']:.0f}GB/s"
        columns.append(f"{device['device']}:{device['name']}{peak}")
    width = max([16] + [len(column) for column in columns])

    def cell(row):
        if row is None:
            return "-"
        if row["status"] == "skipped":
            return "跳过"
        if row["status"] == "error":
            return "出错"
        speedup = row.get("speedup")
        text = "-" if speedup is None else f"{speedup:.2f}x"
        text += " ✓" if row.get("target_met") else " ✗"
        return text

    header = f"{'task':<16}" + "".join(f"{column:>{width}}" for column in columns)
    lines = [header, "-" * len(header)]
    for task in summary["tasks"]:
        lines.append(f"{task:<16}" + "".join(
            f"{cell(_find(summary['rows'], task, device['device'])):>{width}}"
            for device in summary["devices"]
        ))
    totals = summary["totals"]
    geomean_text = "-" if totals["geomean_speedup"] is None else f"{totals['geomean_speedup']:.2f}x"
    lines.append("-" * len(header))
    lines.append(
        f"runs={totals['runs']}  ok={totals['ok']}  failed={totals['failed']}  "
        f"skipped={totals['skipped']}  success@target={totals['successes']}/{totals['ok']}  "
        f"geomean={geomean_text}  tokens={totals['tokens']}"
    )
    return "\n".join(lines)


def _run_one(task_dir: Path, device: str, output: Path, proposer: str, budget: int | None) -> dict:
    started = time.time()
    try:
        task = load_task(task_dir)
        config = Config.from_task(task)
        config.device = device
        config.memory_dir = str(output / "_memory")
        if budget is not None:
            config.budget = budget
        trace_path = output / f"{task.name}@{_slug(device)}.json"
        trace = run(task, config=config, proposer_kind=proposer, out_path=trace_path)
    except Exception as exc:  # noqa: BLE001 - 一个格子的失败不能拖垮整张矩阵
        return _row(task_dir, device, "error", error=f"{type(exc).__name__}: {exc}", started=started)
    metrics = trace.get("metrics") or {}
    return _row(
        task_dir,
        device,
        "ok",
        started=started,
        task=task.name,
        trace=str(trace_path),
        best_ms=metrics.get("best_ms"),
        baseline_ms=metrics.get("baseline_ms"),
        speedup=metrics.get("speedup"),
        target_met=metrics.get("target_met"),
        stop_reason=metrics.get("stop_reason"),
        experiments=trace.get("experiments"),
        probes=trace.get("probes"),
        tokens=trace.get("tokens"),
    )


def _row(task_dir, device, status, *, started=None, error="", **extra) -> dict:
    row = {
        "task": extra.pop("task", Path(task_dir).name),
        "device": device,
        "status": status,
        "error": error,
        "elapsed_s": round(time.time() - started, 2) if started else 0.0,
    }
    row.update(extra)
    return row


def _find(rows: list[dict], task: str, device: str) -> dict | None:
    for row in rows:
        if row["task"] == task and row["device"] == device:
            return row
    return None


def _slug(device: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in device).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="把算子集铺到多块设备上跑")
    parser.add_argument("tasks", nargs="*", default=["inbox"], help="任务目录或父目录")
    parser.add_argument("--devices", default=None, help="逗号分隔，例如 cpu,cuda:0；默认当前最好的设备")
    parser.add_argument("--proposer", default="auto", choices=["auto", "scripted", "llm", "agent"])
    parser.add_argument("--budget", type=int, help="每个格子的候选预算")
    parser.add_argument("--out", default="traces/fleet", help="轨迹与汇总目录")
    parser.add_argument("--jobs", type=int, help="同时跑几块设备（默认等于设备数）")
    parser.add_argument("--json", action="store_true", help="把汇总 JSON 打到 stdout")
    args = parser.parse_args()

    task_dirs = []
    for item in args.tasks:
        task_dirs.extend(list_task_dirs(item))
    if not task_dirs:
        raise SystemExit("没有找到任务目录")
    device_list = (
        [item.strip() for item in args.devices.split(",") if item.strip()]
        if args.devices
        else [best_device()]
    )
    summary = run_fleet(
        task_dirs,
        device_list,
        proposer=args.proposer,
        budget=args.budget,
        out_dir=args.out,
        jobs=args.jobs,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(matrix_text(summary))
        print(f"\nsummary: {Path(args.out) / '_summary.json'}")
    for row in summary["rows"]:
        if row["status"] != "ok":
            print(f"  {row['task']} @ {row['device']}: {row['status']} {row['error']}")


if __name__ == "__main__":
    main()
