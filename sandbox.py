"""候选代码的进程级沙箱。

边界说明（诚实版）：
- 独立子进程 + 独立进程组，超时直接 SIGKILL 整个组；
- 工作目录、HOME、TMPDIR 都指向一次性临时目录；
- 环境变量从白名单重建，不继承 API key / 代理 / 项目路径；
- 设置 RLIMIT_CPU 与 RLIMIT_FSIZE，worker 内再尝试 RLIMIT_AS。

这不是容器级的强隔离：如果候选代码主动 fork 到新会话，或通过网络外连，
进程级限制拦不住。要跑完全不可信的模型生成代码，把 run() 换成容器实现。
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

WORKER = Path(__file__).resolve().with_name("worker.py")
MAX_STDERR = 4000


@dataclass
class SandboxResult:
    data: dict = field(default_factory=dict)
    timed_out: bool = False
    exit_code: int | None = None
    stderr: str = ""
    wall_s: float = 0.0
    command: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.data.get("status", "crash")


def run(job: dict, timeout_s: float = 30.0, memory_mb: int = 0) -> SandboxResult:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kopt-") as tmp:
        root = Path(tmp)
        job_path = root / "job.json"
        out_path = root / "result.json"
        job = {**job, "memory_mb": memory_mb}
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        env = _env(root)
        command = [sys.executable, str(WORKER), str(job_path), str(out_path)]
        proc = subprocess.Popen(
            command,
            cwd=tmp,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_preexec(timeout_s),
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            stdout, stderr = proc.communicate()
        wall = time.perf_counter() - started
        data = _read(out_path)
        if timed_out:
            return SandboxResult(
                data=_synthetic("timeout", f"killed after {timeout_s:.0f}s without a result"),
                timed_out=True,
                exit_code=proc.returncode,
                stderr=_tail(stderr),
                wall_s=wall,
                command=command,
            )
        if data is None:
            detail = _tail(stderr) or _tail(stdout)
            message = f"worker exited with code {proc.returncode}"
            if detail:
                message = f"{message}: {detail}"
            return SandboxResult(
                data=_synthetic("crash", message),
                exit_code=proc.returncode,
                stderr=_tail(stderr),
                wall_s=wall,
                command=command,
            )
        if proc.returncode and proc.returncode < 0:
            data.setdefault("error_kind", "signal")
            data["message"] = f"{data.get('message', '')} (signal {-proc.returncode})".strip()
        return SandboxResult(
            data=data,
            exit_code=proc.returncode,
            stderr=_tail(stderr),
            wall_s=wall,
            command=command,
        )


def _env(root: Path) -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "HOME": str(root),
        "TMPDIR": str(root),
        "TMP": str(root),
        "TEMP": str(root),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LC_ALL": "UTF-8",
        "LANG": "UTF-8",
    }


def _preexec(timeout_s: float):
    def limits() -> None:
        try:
            cpu = max(2, int(timeout_s) + 5)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024, 128 * 1024 * 1024))
        except Exception:
            pass

    return limits


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _read(path: Path) -> dict | None:
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return _synthetic("crash", "result file too large")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "status" not in data:
            return None
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _synthetic(status: str, message: str) -> dict:
    return {
        "status": status,
        "message": message,
        "max_abs_err": None,
        "max_rel_err": None,
        "latency_ms": None,
        "stats": None,
        "profile": None,
        "device": "cpu",
        "error_kind": "sandbox",
    }


def _tail(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[-MAX_STDERR:]


if __name__ == "__main__":  # 手动调试：python sandbox.py job.json
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(run(payload, timeout_s=30).data, ensure_ascii=False, indent=2))
