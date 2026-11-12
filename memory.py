"""跨运行经验：同一算子在同一块卡上，上次跑到了哪里。

单轮内循环只能看到最近 6 轮摘要；换一次运行就全部重来。这里按
`(算子, 设备)` 分文件保存：

- `best`：上次最好的源码和耗时，下次运行先拿它上机（warm start）；
- `lessons`：被采纳过的改法；
- `failures`：反复出现的错误类型（语法、精度、超时…）；
- `dead_ends`：试过但没变快的方向。

A100 上的经验不搬到 M1 上：不同卡的瓶颈不同，跨卡迁移是下一步的事。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import devices

MAX_LESSONS = 6
MAX_FAILURES = 6
MAX_DEAD_ENDS = 5
MAX_CHARS = 220


class MemoryStore:
    def __init__(self, root: str | Path = "memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, task_name: str, spec, shape: str | None = None) -> Path:
        parts = [devices.slug(task_name), devices.slug(spec.name)]
        if shape:
            parts.append(shape)
        return self.root / ("@".join(parts) + ".json")

    def load(self, task_name: str, spec, shape: str | None = None) -> dict:
        try:
            data = json.loads(self.path(task_name, spec, shape).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def update(self, task_name: str, spec, trace: dict, shape: str | None = None) -> dict:
        data = self.load(task_name, spec, shape)
        rounds = trace.get("rounds") or []
        metrics = trace.get("metrics") or {}
        best_name = trace.get("best")
        best_round = next((item for item in rounds if item.get("name") == best_name), {})
        old_best = data.get("best") or {}
        new_best = {
            "name": best_name,
            "hypothesis": _clip(best_round.get("hypothesis") or ""),
            "latency_ms": metrics.get("best_ms", trace.get("best_latency_ms")),
            "baseline_ms": metrics.get("baseline_ms", trace.get("baseline_ms")),
            "speedup": metrics.get("speedup", trace.get("speedup")),
            "source": trace.get("best_source") or "",
        }
        warm_status = next(
            (item.get("status") for item in rounds if item.get("name") == "warm-start"),
            None,
        )
        warm_failed = warm_status not in {None, "accepted", "rejected_slower", "baseline"}
        data["schema_version"] = 1
        data["task"] = task_name
        data["device"] = spec.name
        data["backend"] = spec.backend
        data["shape"] = shape
        data["peak_bw_gbps"] = spec.peak_bw_gbps
        data["runs"] = int(data.get("runs") or 0) + 1
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        # 上次的源码在本设备上崩过/精度不过，就不要再留着；
        # 只是这次没测过它（噪声），或这次没超过它，则继续保留上次的最好版本。
        if best_name == "baseline":
            if old_best and warm_status == "rejected_slower":
                data["best"] = old_best
            else:
                data.pop("best", None)
        elif old_best and not warm_failed and _lat(old_best) <= _lat(new_best):
            data["best"] = old_best
        else:
            data["best"] = new_best

        data["lessons"] = _merge(data.get("lessons"), _lessons(rounds), MAX_LESSONS)
        data["failures"] = _merge_failures(data.get("failures"), rounds)
        data["dead_ends"] = _merge(data.get("dead_ends"), _dead_ends(rounds), MAX_DEAD_ENDS)
        self._write(task_name, spec, data, shape)
        return data

    def _write(self, task_name: str, spec, data: dict, shape: str | None = None) -> None:
        path = self.path(task_name, spec, shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


def signature(task) -> str:
    """按基准输入的形状与 dtype 区分记忆：512x1024 的最优实现不保证对 8x32 最优。"""
    bench = getattr(task, "bench", None) or {}
    args = bench.get("args") or ([bench] if bench else [])
    parts = []
    for arg in args:
        shape = arg.get("shape") or []
        dims = "x".join(str(int(dim)) for dim in shape) if shape else "scalar"
        parts.append(f"{dims}-{_short_dtype(arg.get('dtype') or getattr(task, 'dtype', 'float32'))}")
    text = "-".join(parts) or "default"
    return f"{devices.slug(text)}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:6]}"


def _short_dtype(dtype: str) -> str:
    name = str(dtype).replace("torch.", "")
    return {"float32": "f32", "float16": "f16", "bfloat16": "bf16", "float64": "f64"}.get(name, name)


def summary(data: dict | None) -> dict | None:
    """进 prompt 的压缩视图；不包含 best 源码（warm start 直接读盘）。"""
    if not data:
        return None
    best = data.get("best") or {}
    return {
        "runs": data.get("runs"),
        "best": {
            "hypothesis": best.get("hypothesis") or "",
            "latency_ms": best.get("latency_ms"),
            "baseline_ms": best.get("baseline_ms"),
            "speedup": best.get("speedup"),
        }
        if best
        else None,
        "lessons": (data.get("lessons") or [])[:5],
        "failures": (data.get("failures") or [])[:5],
        "dead_ends": (data.get("dead_ends") or [])[:4],
    }


def best_source(data: dict | None) -> str | None:
    source = ((data or {}).get("best") or {}).get("source")
    return source if isinstance(source, str) and source.strip() else None


def _lessons(rounds: list[dict]) -> list[dict]:
    out = []
    for item in rounds:
        if item.get("status") != "accepted" or item.get("name") in {"baseline", "warm-start"}:
            continue
        out.append(
            {
                "hypothesis": _clip(item.get("hypothesis") or ""),
                "latency_ms": item.get("latency_ms"),
                "speedup": item.get("speedup_vs_baseline"),
            }
        )
    return out


def _dead_ends(rounds: list[dict]) -> list[dict]:
    out = []
    for item in rounds:
        if item.get("status") not in {"rejected_slower", "duplicate"}:
            continue
        if item.get("name") == "warm-start":
            continue
        out.append(
            {
                "hypothesis": _clip(item.get("hypothesis") or ""),
                "status": item.get("status"),
                "latency_ms": item.get("latency_ms"),
            }
        )
    return out


def _merge(old, new, limit: int) -> list[dict]:
    merged = list(new or [])
    for item in old or []:
        if not any(_same(item, other) for other in merged):
            merged.append(item)
    return merged[:limit]


def _merge_failures(old, rounds: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for item in old or []:
        merged[(item.get("status"), item.get("error_kind"))] = dict(item)
    for item in rounds:
        status = item.get("status")
        if status not in {"accuracy_fail", "compile_error", "timeout", "crash", "policy_error"}:
            continue
        key = (status, item.get("error_kind"))
        entry = merged.get(key) or {"status": status, "error_kind": item.get("error_kind"), "count": 0}
        entry["count"] = int(entry.get("count") or 0) + 1
        entry["message"] = _clip(item.get("message") or "")
        entry["hypothesis"] = _clip(item.get("hypothesis") or "")
        merged[key] = entry
    return sorted(merged.values(), key=lambda item: item.get("count") or 0, reverse=True)


def _same(a: dict, b: dict) -> bool:
    return (a.get("hypothesis") or "") == (b.get("hypothesis") or "") and a.get("status") == b.get("status")


def _lat(item: dict) -> float:
    try:
        return float(item.get("latency_ms"))
    except (TypeError, ValueError):
        return float("inf")


def _clip(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text[:MAX_CHARS]
