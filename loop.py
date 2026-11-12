"""把任务、生成器、沙箱和轨迹接起来。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from gates import Runner, best_device, triton_ready
from intake import Task, load_task
from memory import MemoryStore
from proposers import make_proposer
from scheduler import run_schedule
from trace import env_fingerprint, load as load_trace, write as write_trace


@dataclass
class Config:
    budget: int = 8
    best_of: int = 1
    speedup_required: float = 1.1
    target_ms: float | None = None
    patience: int = 4
    repeats: int = 9
    warmup: int = 3
    rtol: float = 1e-4
    atol: float = 1e-5
    timeout_s: float = 30.0
    memory_mb: int = 0
    profile_limit: int = 5
    threads: int | None = None
    max_tokens: int | None = None
    agent_steps: int = 8
    probe_budget: int = 6
    device: str = "cpu"
    peak_bw_gbps: float | None = None
    memory: bool = True
    memory_dir: str = "memory"
    seed: int = 0
    temperature: float = 0.7
    triton_ready: bool = False
    proposer_label: str | None = None

    @classmethod
    def from_task(cls, task: Task) -> "Config":
        return cls(
            budget=task.budget,
            speedup_required=task.speedup_required,
            target_ms=task.target_ms,
            patience=task.patience,
            repeats=task.repeats,
            warmup=task.warmup,
            timeout_s=task.timeout_s,
            memory_mb=task.memory_mb,
            profile_limit=task.profile_limit,
            threads=task.threads,
            probe_budget=task.probe_budget,
            agent_steps=task.agent_steps,
            device="cpu",
            seed=task.seed,
            temperature=task.temperature,
        )

    @classmethod
    def from_trace(cls, trace: dict) -> "Config":
        raw = dict(trace.get("config") or {})
        if not raw:
            raw = {
                "budget": max(1, len(trace.get("rounds", [])) - 1),
                "target_ms": trace.get("target_ms"),
                "repeats": trace.get("repeats", 9),
                "warmup": trace.get("warmup", 3),
                "rtol": trace.get("rtol", 1e-4),
                "atol": trace.get("atol", 1e-5),
                "speedup_required": trace.get("speedup_required", 1.1),
                "device": trace.get("device", "cpu"),
            }
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in allowed})


def run(
    task_dir: str | Path | Task = "inbox/softmax",
    *,
    config: Config | None = None,
    proposer=None,
    proposer_kind: str = "auto",
    out_path: str | Path = "traces/latest.json",
    on_round=None,
) -> dict:
    task = task_dir if isinstance(task_dir, Task) else load_task(task_dir)
    config = config or Config.from_task(task)
    if config.device == "auto":
        config.device = best_device()
    config.triton_ready = triton_ready(config.device)
    if proposer is None:
        proposer, label = make_proposer(proposer_kind, task, config)
    else:
        label = getattr(proposer, "kind", type(proposer).__name__)
    config.proposer_label = label
    runner = Runner(task, config)
    store = MemoryStore(config.memory_dir) if getattr(config, "memory", True) else None
    trace = run_schedule(
        task,
        proposer,
        runner,
        config,
        env=env_fingerprint(config.device),
        on_round=on_round,
        memory_store=store,
    )
    trace["proposer"] = label
    baseline_round = trace["rounds"][0] if trace.get("rounds") else {}
    actual_device = baseline_round.get("device")
    if actual_device:
        trace["device"] = actual_device
        trace["env"]["device_actual"] = actual_device
    if baseline_round.get("threads"):
        trace["env"]["worker_threads"] = baseline_round["threads"]
    write_trace(out_path, trace)
    return trace


def replay(path: str | Path, *, strict: bool = True) -> dict:
    trace = load_trace(path)
    task_dir = trace.get("task_dir") or "inbox/softmax"
    task = load_task(task_dir)
    config = Config.from_trace(trace)
    runner = Runner(task, config)
    mismatches: list[dict] = []
    warnings: list[dict] = []
    checked = 0
    skipped = 0

    for saved in trace.get("rounds", []):
        status = saved.get("status")
        name = saved.get("name", "?")
        source = saved.get("source", "")
        if status == "duplicate":
            skipped += 1
            continue
        if status == "timeout":
            warnings.append({"round": saved.get("round"), "name": name, "warning": "timeout 不重放，避免再次等待"})
            skipped += 1
            continue
        if status == "policy_error":
            from policy import check_source

            kind, message = check_source(source, task.entry)
            checked += 1
            if not kind:
                mismatches.append(
                    {
                        "round": saved.get("round"),
                        "name": name,
                        "expected": status,
                        "got": "policy_pass",
                        "message": "策略当时拒绝，现在通过",
                    }
                )
            continue
        current = runner.check(name, source, profile=False)
        checked += 1
        expected_ok = status in {"baseline", "accepted", "rejected_slower"}
        got_ok = current.status == "ok"
        if expected_ok != got_ok:
            mismatches.append(
                {
                    "round": saved.get("round"),
                    "name": name,
                    "expected": status,
                    "got": current.status,
                    "message": current.message,
                }
            )
            continue
        if not expected_ok and current.status != status:
            mismatches.append(
                {
                    "round": saved.get("round"),
                    "name": name,
                    "expected": status,
                    "got": current.status,
                    "message": current.message,
                }
            )
            continue
        if current.latency_ms and saved.get("latency_ms"):
            ratio = current.latency_ms / saved["latency_ms"]
            if ratio < 1 / 3 or ratio > 3:
                warnings.append(
                    {
                        "round": saved.get("round"),
                        "name": name,
                        "warning": "延迟漂移超过 3×",
                        "saved_ms": saved["latency_ms"],
                        "current_ms": current.latency_ms,
                        "ratio": ratio,
                    }
                )
    report = {
        "trace": str(path),
        "task": trace.get("task"),
        "checked": checked,
        "skipped": skipped,
        "ok": not mismatches,
        "mismatches": mismatches,
        "warnings": warnings,
        "best": trace.get("best"),
        "stop": trace.get("stop"),
    }
    if mismatches and strict:
        raise RuntimeError(
            "replay mismatch: "
            + json.dumps(mismatches[:3], ensure_ascii=False)
        )
    return report


def config_to_dict(config: Config) -> dict:
    return asdict(config)
