"""轨迹的落盘与环境指纹。

可复现的前提不是“延迟小数点后三位一样”，而是：
- 知道这组数字是在什么机器、什么版本、多少线程上跑的；
- 知道每一轮源码、门禁结果和反馈；
- 能在同一台机器上重放，至少把正确性判定复现出来。
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path


def env_fingerprint(device: str = "cpu") -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device_requested": device,
        "cwd": str(Path.cwd()),
        "git_sha": _git_sha(),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": bool(
                    getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                ),
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
            }
        )
    except Exception:
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:
        pass
    return info


def write(path: str | Path, trace: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None
