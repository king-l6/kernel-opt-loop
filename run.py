"""跑一个任务，或重放已有轨迹。"""

from __future__ import annotations

import argparse
import json

import devices
from intake import load_task
from loop import Config, replay, run


def main() -> None:
    parser = argparse.ArgumentParser(description="读入 reference.py，跑算子优化闭环")
    parser.add_argument("task", nargs="?", default="inbox/softmax")
    parser.add_argument("--proposer", default="auto", choices=["auto", "scripted", "llm", "agent"])
    parser.add_argument("--best-of", type=int, help="每轮并行生成几个候选")
    parser.add_argument("--budget", type=int, help="最多消耗几个候选")
    parser.add_argument("--target", type=float, help="耗时目标（ms）")
    parser.add_argument("--out", default="traces/latest.json")
    parser.add_argument("--device", help="cpu / mps / cuda / cuda:1 / auto")
    parser.add_argument("--peak-bw", type=float, help="覆盖设备的纸面峰值带宽（GB/s），用于未登记的卡")
    parser.add_argument("--timeout", type=float, help="单次沙箱执行的超时秒数")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-tokens", type=int, help="整条轨迹的 token 预算")
    parser.add_argument("--agent-steps", type=int, help="Agent 每轮最多几步（含试跑）")
    parser.add_argument("--probe-budget", type=int, help="Agent 整条运行最多试跑几次")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--replay", help="重放某条轨迹 JSON，核对每轮状态")
    parser.add_argument("--lenient", action="store_true", help="重放时只报告不一致，不抛错")
    parser.add_argument("--json", action="store_true", help="把轨迹 JSON 打到 stdout")
    parser.add_argument("--progress", action="store_true", help="每轮输出一行 PROGRESS JSON，供页面实时显示")
    args = parser.parse_args()

    if args.replay:
        report = replay(args.replay, strict=not args.lenient)
        print(
            f"replay {'ok' if report['ok'] else 'mismatch'}  "
            f"checked={report['checked']} skipped={report['skipped']} "
            f"warnings={len(report['warnings'])}"
        )
        for item in report["mismatches"]:
            print(f"  mismatch round {item['round']} {item['name']}: {item['expected']} -> {item['got']}")
        for item in report["warnings"]:
            print(f"  warning round {item['round']} {item['name']}: {item['warning']}")
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    task = load_task(args.task)
    config = Config.from_task(task)
    if args.best_of is not None:
        config.best_of = args.best_of
    if args.budget is not None:
        config.budget = args.budget
    if args.target is not None:
        config.target_ms = args.target
    if args.device is not None:
        if args.device != "auto" and not devices.available(args.device):
            raise SystemExit(f"设备不可用：{args.device}（不会悄悄退回 CPU，换 --device 或先确认驱动）")
        config.device = args.device
    if args.peak_bw is not None:
        config.peak_bw_gbps = args.peak_bw
    if args.timeout is not None:
        config.timeout_s = args.timeout
    if args.seed is not None:
        config.seed = args.seed
    if args.threads is not None:
        config.threads = args.threads
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.agent_steps is not None:
        config.agent_steps = args.agent_steps
    if args.probe_budget is not None:
        config.probe_budget = args.probe_budget
    if args.temperature is not None:
        config.temperature = args.temperature

    on_round = _progress if args.progress and not args.json else None
    trace = run(
        task,
        config=config,
        proposer_kind=args.proposer,
        out_path=args.out,
        on_round=on_round,
    )
    if args.json:
        print(json.dumps(trace, ensure_ascii=False, indent=2))
        return
    _print_table(trace)
    print(f"\nproposer: {trace.get('proposer')}")
    print(f"best: {trace['best']}  {trace['metrics']['best_ms']:.3f} ms  "
          f"speedup={trace['metrics']['speedup']:.2f}x  target_met={trace['metrics']['target_met']}")
    print(f"stop: {trace['stop']}  ({trace['stop_reason']})")
    print(f"trace: {args.out}")


def _progress(event: dict) -> None:
    print("PROGRESS " + json.dumps(event, ensure_ascii=False), flush=True)


def _print_table(trace: dict) -> None:
    info = trace.get("device_info") or {}
    peak = info.get("peak_bw_gbps")
    device = info.get("name") or trace["device"]
    if peak:
        device += f" ({peak:.0f} GB/s)"
    print(
        f"{trace['reference_file']}  {trace['card']}  "
        f"shape={tuple(trace['shape'])}  device={device}  "
        f"proposer={trace.get('proposer')}"
    )
    print(f"{'round':<6}{'name':<18}{'status':<18}{'latency_ms':>12}{'speedup':>9}  message")
    for item in trace["rounds"]:
        latency = "-" if item["latency_ms"] is None else f"{item['latency_ms']:.3f}"
        speedup = item.get("speedup_vs_baseline")
        speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
        message = (item["message"] or "").replace("\n", " ")
        print(
            f"{item['round']:<6}{item['name']:<18}{item['status']:<18}"
            f"{latency:>12}{speedup_text:>9}  {message}"
        )


if __name__ == "__main__":
    main()
