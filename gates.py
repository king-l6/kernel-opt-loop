"""编译、精度、测速三道门。所有候选代码都通过沙箱子进程执行。"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch

import sandbox


@dataclass
class Eval:
    status: str
    message: str
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    latency_ms: float | None = None
    stats: dict | None = None
    profile: list | None = None
    error_kind: str = ""
    device: str = "cpu"
    threads: int | None = None
    wall_s: float = 0.0
    exit_code: int | None = None
    stderr: str = ""

    @classmethod
    def from_result(cls, result: sandbox.SandboxResult) -> "Eval":
        data = result.data
        return cls(
            status=data.get("status", "crash"),
            message=data.get("message", ""),
            max_abs_err=data.get("max_abs_err"),
            max_rel_err=data.get("max_rel_err"),
            latency_ms=data.get("latency_ms"),
            stats=data.get("stats"),
            profile=data.get("profile"),
            error_kind=data.get("error_kind", "") or ("sandbox" if result.timed_out else ""),
            device=data.get("device", "cpu"),
            threads=data.get("threads"),
            wall_s=result.wall_s,
            exit_code=result.exit_code,
            stderr=result.stderr,
        )


class Runner:
    """把任务和配置翻译成 worker job，并收回 Eval。"""

    def __init__(self, task, config):
        self.task = task
        self.config = config

    def check(self, name: str, source: str, profile: bool = False) -> Eval:
        job = self._job(name, source, mode="eval", profile=profile)
        return Eval.from_result(
            sandbox.run(job, timeout_s=self.config.timeout_s, memory_mb=self.config.memory_mb)
        )

    def profile(self, name: str, source: str) -> list:
        job = self._job(name, source, mode="profile", profile=False)
        result = sandbox.run(job, timeout_s=self.config.timeout_s, memory_mb=self.config.memory_mb)
        if result.status != "ok":
            return []
        return result.data.get("profile") or []

    def _job(self, name: str, source: str, mode: str, profile: bool) -> dict:
        return {
            "name": name,
            "mode": mode,
            "entry": self.task.entry,
            "reference": self.task.source,
            "source": source,
            "accuracy": self.task.accuracy,
            "bench": self.task.bench,
            "dtype": self.task.dtype,
            "device": str(getattr(self.config, "device", "cpu")),
            "rtol": self.config.rtol,
            "atol": self.config.atol,
            "warmup": self.config.warmup,
            "repeats": self.config.repeats,
            "seed": self.config.seed,
            "threads": self.config.threads,
            "profile_limit": self.config.profile_limit,
            "profile": profile,
        }


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def triton_ready(device: str | None = None) -> bool:
    device = device or best_device()
    if device != "cuda" or not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("triton") is not None
