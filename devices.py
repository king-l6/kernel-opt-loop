"""设备反馈：把“这块卡跑到多少”变成生成器能用的结构化信息。

同一份算子在不同卡上的瓶颈不一样，只给毫秒数不够。这里在父进程做三件事：

1. 认出这次用的是哪块卡，拿到它的纸面峰值带宽；
2. 用子进程回报的最小 I/O 字节数，算出这次算子的有效带宽；
3. 给出「单次 I/O 下界」和剩余空间，下一轮就知道该压 pass 还是压字节数。

子进程只负责测，不负责判断；换成别的卡时，只要换 `DeviceSpec`，门禁和反馈
流程不用动。
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field

# 纸面峰值带宽（GB/s）。只收公开规格，查不到就留空，宁可不说也不编。
# 名称按小写子串匹配，长键优先，所以 "apple m1 pro" 不会被 "apple m1" 抢走。
_SPECS: dict[str, float] = {
    # Apple Silicon（统一内存，全芯片带宽）
    "apple m1 ultra": 800.0,
    "apple m1 max": 400.0,
    "apple m1 pro": 200.0,
    "apple m1": 68.25,
    "apple m2 ultra": 800.0,
    "apple m2 max": 400.0,
    "apple m2 pro": 200.0,
    "apple m2": 100.0,
    "apple m3 max": 400.0,
    "apple m3 pro": 150.0,
    "apple m3": 100.0,
    "apple m4 max": 546.0,
    "apple m4 pro": 273.0,
    "apple m4": 120.0,
    # NVIDIA 数据中心卡
    "h100 sxm": 3350.0,
    "h100 pcie": 2000.0,
    "h200": 4800.0,
    "a100-sxm4-80gb": 2039.0,
    "a100-sxm4-40gb": 1555.0,
    "a100-pcie-80gb": 1935.0,
    "a100-pcie-40gb": 1555.0,
    "a800": 2039.0,
    "a10g": 600.0,
    "a10": 600.0,
    "a40": 696.0,
    "l40s": 864.0,
    "l40": 864.0,
    "l4": 300.0,
    "v100": 900.0,
    "t4": 320.0,
    # NVIDIA 消费卡
    "rtx 4090": 1008.0,
    "rtx 4080": 717.0,
    "rtx 3090": 936.0,
    "rtx 3080": 760.0,
    "rtx 2080 ti": 616.0,
    "rtx a6000": 768.0,
}

_ORDERED_SPECS = sorted(_SPECS.items(), key=lambda item: len(item[0]), reverse=True)


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    backend: str
    name: str
    peak_bw_gbps: float | None = None
    memory_gb: float | None = None
    source: str = "unknown"  # spec | override | unknown
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def spec_for(task, config, device: str | None = None) -> DeviceSpec:
    """任务 + 配置 → 这次运行用的设备描述。CLI 的 --peak-bw 优先于任务声明。

    `device` 用于子进程实际回退后的结果（例如要了 cuda 但只拿到 cpu）。
    """
    override = dict(getattr(task, "device_spec", None) or {})
    peak = getattr(config, "peak_bw_gbps", None)
    if peak:
        override["peak_bw_gbps"] = peak
    return detect(device or str(getattr(config, "device", "cpu")), override)


def detect(device: str = "cpu", override: dict | None = None) -> DeviceSpec:
    backend, index = _split_device(device)
    name = ""
    memory_gb = None
    detail: dict = {}
    if backend == "cuda":
        name, memory_gb, detail = _cuda_info(index)
    elif backend == "mps":
        name = f"{local_cpu_name()} (MPS)"
        memory_gb = total_memory_gb()
    else:
        backend = "cpu"
        name = local_cpu_name()
        memory_gb = total_memory_gb()

    if override and override.get("name"):
        name = str(override["name"])
    peak = _lookup_peak(name)
    source = "spec" if peak else "unknown"
    if override:
        if _positive(override.get("peak_bw_gbps")):
            peak = float(override["peak_bw_gbps"])
            source = "override"
        if _positive(override.get("memory_gb")):
            memory_gb = float(override["memory_gb"])
    return DeviceSpec(
        key=device or "cpu",
        backend=backend,
        name=name or device or "cpu",
        peak_bw_gbps=peak,
        memory_gb=memory_gb,
        source=source,
        detail=detail,
    )


def available(device: str) -> bool:
    """显式指定的卡不存在时返回 False，绝不悄悄退回到 CPU。"""
    backend, _ = _split_device(device)
    if backend == "cpu":
        return True
    try:
        import torch
    except Exception:
        return False
    if backend == "cuda":
        return bool(torch.cuda.is_available())
    if backend == "mps":
        backend_mod = getattr(torch.backends, "mps", None)
        return bool(backend_mod and backend_mod.is_available())
    return False


def label(spec: DeviceSpec) -> str:
    """按“卡 + 峰值”给人看的短标签，用来区分同一算子的不同设备结果。"""
    peak = "" if spec.peak_bw_gbps is None else f" {spec.peak_bw_gbps:.0f}GB/s"
    return f"{spec.name}{peak}"


def analyze(spec: DeviceSpec, latency_ms: float | None, io_bytes: int | None) -> dict:
    """把一次测速解释成「离这块卡的单次 I/O 下界还有多远」。

    io_bytes 是最小 I/O（输入 + 输出各一遍），不是真实 DRAM 流量：
    多 pass 的中间张量不计入，所以这里得到的是**乐观下界**，不是利用率。
    """
    result = {
        "device": spec.name,
        "backend": spec.backend,
        "peak_bw_gbps": spec.peak_bw_gbps,
        "peak_source": spec.source,
        "latency_ms": latency_ms,
        "io_bytes": io_bytes,
        "gbps": None,
        "bw_pct": None,
        "floor_ms": None,
        "io_headroom": None,
        "cache_resident": False,
        "advice": [],
    }
    if not latency_ms or latency_ms <= 0 or not io_bytes or io_bytes <= 0:
        return result

    gbps = io_bytes / (latency_ms / 1000.0) / 1e9
    result["gbps"] = gbps
    peak = spec.peak_bw_gbps
    if not peak:
        result["advice"] = [
            f"{spec.name} 没登记峰值带宽，只报有效带宽；在 task.json 的 device_spec "
            f"里补 peak_bw_gbps，或加 --peak-bw，就能算出单次 I/O 下界。"
        ]
        return result

    floor_ms = io_bytes / (peak * 1e9) * 1000.0
    result["floor_ms"] = floor_ms
    result["bw_pct"] = gbps / peak
    result["io_headroom"] = latency_ms / floor_ms
    result["cache_resident"] = result["bw_pct"] > 1.15
    result["advice"] = _advice(spec, result)
    return result


def describe(hardware: dict | None) -> str:
    """一行中文，直接进反馈文本。"""
    if not hardware:
        return ""
    device = hardware.get("device") or "当前设备"
    gbps = hardware.get("gbps")
    io_bytes = hardware.get("io_bytes")
    if gbps is None:
        return f"{device}：这次没有采到 I/O 字节数。"
    size = "" if not io_bytes else f"（最小 I/O {io_bytes / 1e6:.2f} MB）"
    peak = hardware.get("peak_bw_gbps")
    if not peak:
        return f"{device}：有效带宽 {gbps:.1f} GB/s{size}，峰值未登记。"
    pct = hardware.get("bw_pct") or 0.0
    text = (
        f"{device}：有效带宽 {gbps:.1f} GB/s{size}，"
        f"纸面峰值 {peak:.0f} GB/s 的 {pct:.0%}，"
        f"单次 I/O 下界 {hardware['floor_ms']:.3f} ms。"
    )
    if hardware.get("cache_resident"):
        text += "工作集偏小，结果可能命中缓存，带宽数字不代表 DRAM 上限。"
    return text


# --------------------------------------------------------------------------
# 内部
# --------------------------------------------------------------------------


def _advice(spec: DeviceSpec, h: dict) -> list[str]:
    pct = h["bw_pct"]
    floor = h["floor_ms"]
    latency = h["latency_ms"]
    headroom = h["io_headroom"]
    if h["cache_resident"]:
        return [
            f"工作集只有 {h['io_bytes'] / 1e6:.2f} MB，按最小 I/O 算出的带宽已超过纸面峰值"
            f"（{pct:.0%}），多半落在缓存里；这组 shape 不能代表 DRAM 上限，"
            f"换更大的 shape，或看 profiler 里的实际访存计数。"
        ]
    if pct >= 0.6:
        return [
            f"已用掉 {pct:.0%} 纸面带宽，离单次 I/O 下界 {floor:.3f} ms 不远；"
            f"再快主要靠减少字节数：融合 pass、原地写、低精度、去掉中间张量。"
        ]
    if pct >= 0.2:
        return [
            f"按最小 I/O 只用到 {pct:.0%} 峰值带宽；先压 pass 数和中间张量，"
            f"再看访存是否合并。下界 {floor:.3f} ms，当前 {latency:.3f} ms，"
            f"乐观上限 {headroom:.1f}×（未计中间张量）。"
        ]
    return [
        f"按最小 I/O 只有 {pct:.0%} 峰值带宽，离下界 {floor:.3f} ms 还有 {headroom:.1f}×"
        f"（未计中间张量）；先换成整块张量运算，避免逐行 Python 循环。"
    ]


def _lookup_peak(name: str) -> float | None:
    normalized = " ".join(name.lower().split())
    for key, peak in _ORDERED_SPECS:
        if key in normalized:
            return peak
    return None


def _split_device(device: str) -> tuple[str, int | None]:
    parts = str(device or "cpu").split(":")
    backend = parts[0].strip().lower() or "cpu"
    index = None
    if len(parts) > 1 and parts[1].strip().isdigit():
        index = int(parts[1])
    return backend, index


def _cuda_info(index: int | None) -> tuple[str, float | None, dict]:
    try:
        import torch

        if not torch.cuda.is_available():
            return "", None, {}
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        detail = {
            "index": index,
            "sm_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
        }
        return props.name, round(props.total_memory / 1e9, 1), detail
    except Exception:
        return "", None, {}


@functools.lru_cache(maxsize=1)
def local_cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "cpu"


@functools.lru_cache(maxsize=1)
def total_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return round(pages * size / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _positive(value) -> bool:
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False
