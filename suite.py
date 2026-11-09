"""把一个目录下的任务逐个跑完，再汇总成一张表。

测速必须独占机器，所以任务之间串行；并发的部分在生成侧（best-of-N 并行采样）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from intake import list_task_dirs, load_task
from loop import Config, run
from metrics import report_text, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="跑一个任务集并汇总")
    parser.add_argument("root", nargs="?", default="inbox", help="任务目录或其父目录")
    parser.add_argument("--proposer", default="auto", choices=["auto", "scripted", "llm"])
    parser.add_argument("--out", default="traces/suite", help="每条轨迹的落盘目录")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--best-of", type=int)
    parser.add_argument("--target", type=float)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    task_dirs = list_task_dirs(args.root)
    if not task_dirs:
        raise SystemExit(f"{args.root} 下没有 task.json")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    failures = []
    for task_dir in task_dirs:
        started = time.time()
        print(f"[{task_dir.name}] 开始", flush=True)
        try:
            task = load_task(task_dir)
            config = Config.from_task(task)
            if args.budget is not None:
                config.budget = args.budget
            if args.best_of is not None:
                config.best_of = args.best_of
            if args.target is not None:
                config.target_ms = args.target
            if args.device is not None:
                config.device = args.device
            if args.timeout is not None:
                config.timeout_s = args.timeout
            trace = run(
                task,
                config=config,
                proposer_kind=args.proposer,
                out_path=out_dir / f"{task.name}.json",
            )
            traces.append(trace)
            metrics = trace["metrics"]
            print(
                f"[{task_dir.name}] best={trace['best']} "
                f"{metrics['best_ms']:.3f}ms speedup={metrics['speedup']:.2f}x "
                f"rounds={metrics['rounds']} ({time.time() - started:.1f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 一个任务失败不应中断整个任务集
            failures.append({"task": task_dir.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{task_dir.name}] 失败：{type(exc).__name__}: {exc}", flush=True)
    if failures:
        (out_dir / "_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not traces:
        raise SystemExit("没有任何任务成功")
    summary = summarize(traces)
    summary["failures"] = failures
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(report_text(summary))
        if failures:
            print(f"\n失败 {len(failures)} 个任务，详见 {out_dir / '_failures.json'}")


if __name__ == "__main__":
    main()
