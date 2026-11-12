"""内循环工具：让生成器在提交前自己试跑候选。

Agent 每轮拿到的不是一个只能返回源码的函数，而是一个受预算约束的沙箱探针：
- 先做静态策略检查，语法/越界源码不消耗真机预算；
- 通过后交给和正式门禁完全相同的 Runner，返回状态、错误、耗时、profile；
- 同一份源码（AST 相同）只跑一次，命中缓存；
- probe 预算独立于提交预算，防止 Agent 无限试跑。

这里不更新 best，也不写轨迹；最终是否采纳仍由 scheduler 的门禁决定。
"""

from __future__ import annotations

from typing import Callable

import devices
from policy import check_source
from proposers import source_signature


def _tail(text: str | None, limit: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[-limit:]


class ProbeSession:
    """一次运行内共享的探针预算与缓存。"""

    def __init__(
        self,
        task,
        runner,
        config,
        on_event: Callable[[dict], None] | None = None,
        spec=None,
    ) -> None:
        self.task = task
        self.runner = runner
        self.spec = spec or devices.spec_for(task, config)
        self.limit = max(0, int(getattr(config, "probe_budget", 0) or 0))
        self.used = 0
        self.cache: dict[str, dict] = {}
        self.on_event = on_event
        self.best_ms: float | None = None
        self.round_no = 0

    @property
    def left(self) -> int:
        return max(0, self.limit - self.used)

    def probe(self, source, profile: bool = False, name: str = "probe") -> dict:
        if not isinstance(source, str) or not source.strip():
            return self._result("policy_error", "源码为空", "policy")
        kind, message = check_source(source, self.task.entry)
        if kind == "syntax":
            return self._result("compile_error", message, "syntax")
        if kind == "policy":
            return self._result("policy_error", message, "policy")

        signature = source_signature(source)
        cached = self.cache.get(signature)
        if cached is not None and (cached.get("profile") or not profile):
            return {**cached, "cached": True}

        if self.used >= self.limit:
            return self._result("probe_budget", "probe 预算用完，直接提交你目前最好的一版。", "budget")

        self.used += 1
        label = f"{name}-{self.used}"
        result = self.runner.check(label, source, profile=bool(profile))
        io_bytes = getattr(result, "io_bytes", None)
        hardware = devices.analyze(self.spec, result.latency_ms, io_bytes)
        payload = {
            "status": result.status,
            "message": result.message,
            "latency_ms": result.latency_ms,
            "max_abs_err": result.max_abs_err,
            "max_rel_err": result.max_rel_err,
            "stats": result.stats,
            "profile": result.profile or [],
            "error_kind": result.error_kind,
            "stderr": _tail(result.stderr),
            "device": result.device,
            "io_bytes": io_bytes,
            "hardware": hardware,
            "profile_requested": bool(profile),
            "probe_left": self.left,
            "cached": False,
        }
        self.cache[signature] = payload
        self._emit(payload, label)
        return payload

    def _emit(self, payload: dict, label: str) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(
                {
                    "kind": "probe",
                    "round": self.round_no,
                    "name": label,
                    "title": "试跑",
                    "status": payload["status"],
                    "message": payload.get("message") or "",
                    "latency_ms": payload.get("latency_ms"),
                    "speedup_vs_baseline": None,
                    "best_ms": self.best_ms,
                    "error_kind": payload.get("error_kind", ""),
                    "gbps": (payload.get("hardware") or {}).get("gbps"),
                }
            )
        except Exception:  # noqa: BLE001 - 进度展示是旁路
            pass

    def _result(self, status: str, message: str, error_kind: str) -> dict:
        return {
            "status": status,
            "message": message,
            "latency_ms": None,
            "max_abs_err": None,
            "max_rel_err": None,
            "stats": None,
            "profile": [],
            "error_kind": error_kind,
            "stderr": "",
            "device": "",
            "io_bytes": None,
            "hardware": None,
            "profile_requested": False,
            "probe_left": self.left,
            "cached": False,
        }
