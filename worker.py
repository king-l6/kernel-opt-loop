"""子进程执行器：编译、精度、测速、profile 都发生在这里。

父进程不 import、不 exec 任何候选代码。这个文件只通过两个参数工作：
  1. job JSON 的路径
  2. 结果 JSON 的写出路径

候选代码即使死循环、崩溃、调用 sys.exit，也只影响这个子进程。
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import traceback


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: worker.py JOB OUT")
    job_path, out_path = sys.argv[1], sys.argv[2]
    try:
        with open(job_path, encoding="utf-8") as fh:
            job = json.load(fh)
        result = execute(job)
    except SystemExit as exc:
        result = _fail("crash", f"candidate called sys.exit({exc.code})", "runtime")
    except BaseException as exc:  # noqa: BLE001 - 子进程边界必须兜底
        result = _fail(
            "crash",
            f"{type(exc).__name__}: {exc}",
            "runtime",
            detail=traceback.format_exc(limit=6),
        )
    try:
        import torch

        result.setdefault("threads", torch.get_num_threads())
    except Exception:
        pass
    _write(out_path, result)


def execute(job: dict) -> dict:
    import torch

    torch.manual_seed(int(job.get("seed", 0)))
    _limit_memory(int(job.get("memory_mb") or 0))
    threads = job.get("threads")
    if threads:
        try:
            torch.set_num_threads(int(threads))
        except Exception:
            pass
    device = _device(torch, job.get("device") or "cpu")
    entry = job["entry"]
    source = job.get("source") or ""

    if job.get("mode") == "profile":
        fn, kind, err = _load(torch, source, entry, "<candidate>")
        if fn is None:
            return _fail("compile_error", err, kind)
        try:
            args = _make_args(torch, job.get("bench", {}), device, int(job.get("seed", 1)))
            profile = _profile(torch, fn, args, device, int(job.get("profile_limit", 5)))
        except Exception as exc:
            return _fail("crash", f"{type(exc).__name__}: {exc}", "runtime")
        return _ok(profile=profile, device=device.type)

    ref_fn, kind, err = _load(torch, job["reference"], entry, "<reference>")
    if ref_fn is None:
        return _fail("crash", f"reference load failed: {err}", kind)

    fn, kind, err = _load(torch, source, entry, "<candidate>")
    if fn is None:
        return _fail("compile_error", err, kind)

    accuracy = _check_accuracy(torch, fn, ref_fn, job, device)
    if accuracy["status"] != "ok":
        return accuracy

    latency, stats = _bench(torch, fn, job, device)
    profile = None
    if job.get("profile"):
        profile = _profile(
            torch,
            fn,
            _make_args(torch, job.get("bench", {}), device, int(job.get("seed", 1))),
            device,
            int(job.get("profile_limit", 5)),
        )
    return _ok(
        latency_ms=latency,
        stats=stats,
        profile=profile,
        device=device.type,
        max_abs_err=accuracy.get("max_abs_err"),
        max_rel_err=accuracy.get("max_rel_err"),
    )


# --------------------------------------------------------------------------
# 加载与校验
# --------------------------------------------------------------------------


def _load(torch, source: str, entry: str, filename: str):
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as exc:
        return None, "syntax", f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except ValueError as exc:
        return None, "syntax", f"ValueError: {exc}"
    namespace = {"torch": torch, "__name__": filename.strip("<>")}
    try:
        exec(code, namespace, namespace)
    except Exception as exc:  # noqa: BLE001 - 候选代码的运行错误
        return None, "load", f"{type(exc).__name__}: {exc}"
    fn = namespace.get(entry)
    if not callable(fn):
        return None, "load", f"candidate must define {entry}(...)"
    return fn, "", ""


def _check_accuracy(torch, fn, ref_fn, job: dict, device) -> dict:
    rtol = float(job.get("rtol", 1e-4))
    atol = float(job.get("atol", 1e-5))
    max_abs = 0.0
    max_rel = 0.0
    cases = job.get("accuracy") or []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            try:
                args = _make_args(torch, case, device, int(case.get("seed", index)))
            except Exception as exc:
                return _fail("crash", f"bad accuracy case {index}: {exc}", "input")
            try:
                golden = ref_fn(*args)
            except Exception as exc:
                return _fail("crash", f"reference failed on case {index}: {exc}", "reference")
            try:
                out = fn(*args)
            except Exception as exc:
                return _fail(
                    "accuracy_fail",
                    f"case {index}: {type(exc).__name__}: {exc}",
                    "runtime",
                    max_abs_err=_num(max_abs),
                    max_rel_err=_num(max_rel),
                )
            if not isinstance(out, torch.Tensor):
                return _fail(
                    "accuracy_fail",
                    f"case {index}: returned {type(out).__name__}, not a Tensor",
                    "interface",
                    max_abs_err=_num(max_abs),
                    max_rel_err=_num(max_rel),
                )
            if out.shape != golden.shape:
                return _fail(
                    "accuracy_fail",
                    f"case {index}: shape {tuple(out.shape)} != {tuple(golden.shape)}",
                    "shape",
                    max_abs_err=_num(max_abs),
                    max_rel_err=_num(max_rel),
                )
            if out.dtype != golden.dtype:
                return _fail(
                    "accuracy_fail",
                    f"case {index}: dtype {out.dtype} != {golden.dtype}",
                    "dtype",
                    max_abs_err=_num(max_abs),
                    max_rel_err=_num(max_rel),
                )
            finite = bool(torch.isfinite(out).all().item())
            if not finite:
                return _fail(
                    "accuracy_fail",
                    f"case {index}: non-finite output",
                    "non_finite",
                    max_abs_err=_num(max_abs),
                    max_rel_err=_num(max_rel),
                )
            diff = (out - golden).abs()
            abs_err = float(diff.max().item()) if diff.numel() else 0.0
            denom = golden.abs().clamp_min(1e-12)
            rel_err = float((diff / denom).max().item()) if diff.numel() else 0.0
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if not bool(torch.allclose(out, golden, rtol=rtol, atol=atol)):
                return _fail(
                    "accuracy_fail",
                    f"case {index}: max_abs_err={abs_err:.3e}, max_rel_err={rel_err:.3e}",
                    "mismatch",
                    max_abs_err=_num(max_abs),
                    max_rel_err=_num(max_rel),
                )
    return {
        "status": "ok",
        "message": "accuracy passed",
        "max_abs_err": _num(max_abs),
        "max_rel_err": _num(max_rel),
        "latency_ms": None,
        "stats": None,
        "profile": None,
        "device": device.type,
        "error_kind": "",
    }


# --------------------------------------------------------------------------
# 输入、测速、profile
# --------------------------------------------------------------------------


def _make_args(torch, case: dict, device, seed: int):
    specs = case.get("args")
    if specs is None:
        specs = [case]
    dtype_name = case.get("dtype") or "float32"
    dtype = getattr(torch, dtype_name)
    args = []
    for index, spec in enumerate(specs):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(spec.get("seed", seed + index)))
        shape = tuple(int(x) for x in spec["shape"])
        dist = spec.get("dist", "randn")
        item_dtype = getattr(torch, spec.get("dtype", dtype_name))
        if dist == "zeros":
            tensor = torch.zeros(*shape, dtype=item_dtype)
        elif dist == "ones":
            tensor = torch.ones(*shape, dtype=item_dtype)
        elif dist in {"full", "constant"}:
            tensor = torch.full(shape, float(spec.get("value", 0.0)), dtype=item_dtype)
        elif dist == "uniform":
            scale = float(spec.get("scale", 1.0))
            tensor = (torch.rand(*shape, generator=gen, dtype=item_dtype) - 0.5) * (2.0 * scale)
        elif dist == "randint":
            low = int(spec.get("low", 0))
            high = int(spec.get("high", 10))
            tensor = torch.randint(low, high, shape, generator=gen, dtype=torch.int64)
            tensor = tensor.to(item_dtype)
        else:  # randn
            scale = float(spec.get("scale", 1.0))
            tensor = torch.randn(*shape, generator=gen, dtype=item_dtype) * scale
        if spec.get("transpose"):
            tensor = tensor.transpose(-1, -2)
        args.append(tensor.to(device))
    return args


def _bench(torch, fn, job: dict, device):
    args = _make_args(torch, job.get("bench", {}), device, int(job.get("seed", 1)))
    warmup = int(job.get("warmup", 3))
    repeats = int(job.get("repeats", 9))
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args)
        _sync(torch, device)
        samples = []
        cuda = device.type == "cuda"
        for _ in range(repeats):
            if cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(*args)
                end.record()
                torch.cuda.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                begin = time.perf_counter()
                fn(*args)
                _sync(torch, device)
                samples.append((time.perf_counter() - begin) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return median, {
        "p50_ms": median,
        "p90_ms": samples[min(len(samples) - 1, int(math.ceil(0.9 * len(samples))) - 1)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": mean,
        "stdev_ms": stdev,
        "cv": (stdev / mean) if mean else None,
        "samples": len(samples),
    }


def _profile(torch, fn, args, device, limit: int):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities) as prof:
            fn(*args)
            _sync(torch, device)
    except Exception:
        return []
    key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    rows = sorted(prof.key_averages(), key=lambda e: getattr(e, key, 0), reverse=True)
    total = sum(max(0.0, float(getattr(e, key, 0))) for e in rows) or 1.0
    result = []
    for event in rows[:limit]:
        us = max(0.0, float(getattr(event, key, 0)))
        result.append(
            {
                "name": event.key,
                "us": us,
                "pct": 100.0 * us / total,
            }
        )
    return result


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _device(torch, name: str):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _limit_memory(mb: int) -> None:
    if not mb:
        return
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = int(mb) * 1024 * 1024
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except Exception:
        pass


# --------------------------------------------------------------------------
# 结果
# --------------------------------------------------------------------------


def _ok(**kwargs) -> dict:
    base = {
        "status": "ok",
        "message": "accuracy passed",
        "max_abs_err": None,
        "max_rel_err": None,
        "latency_ms": None,
        "stats": None,
        "profile": None,
        "device": "cpu",
        "error_kind": "",
    }
    base.update(kwargs)
    return base


def _fail(status: str, message: str, kind: str, **extra) -> dict:
    base = {
        "status": status,
        "message": message,
        "max_abs_err": None,
        "max_rel_err": None,
        "latency_ms": None,
        "stats": None,
        "profile": None,
        "device": "cpu",
        "error_kind": kind,
    }
    base.update(extra)
    return base


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write(path: str, payload: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, default=str)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
