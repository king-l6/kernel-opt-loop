"""讲解页 + 可真实执行的本地控制台。

只读接口：
- GET  /api/tasks                    列出 inbox 下的任务
- GET  /api/loop/run?path=...&diff=1 读一条轨迹（diff=1 时附带每轮和原版的源码差异）
- GET  /api/traces                   列出可切换的轨迹
- GET  /api/report                   每个任务取最新一条轨迹做汇总

执行接口：
- POST /api/run                      启动一次真实运行 {"task","proposer","best_of","budget"}
- POST /api/run/stop                 停止运行中的任务 {"id"}，先 SIGTERM 再 SIGKILL
- GET  /api/run?id=...               轮询运行状态、stdout、逐轮进度
- GET  /api/runs                     最近的运行记录

服务只绑 127.0.0.1；POST 只接受 inbox 里的任务名和固定的 proposer 白名单。
"""

from __future__ import annotations

import difflib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from metrics import is_trace, summarize

ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "traces"
WEB_RUN_DIR = TRACE_DIR / "web"
PORT = 8765
MAX_LINES = 2000
MAX_BODY = 64 * 1024
PROPOSERS = {"auto", "scripted", "llm"}
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


@dataclass
class RunState:
    id: str
    task: str
    proposer: str
    state: str = "running"  # running | done | error
    exit_code: int | None = None
    started: float = field(default_factory=time.time)
    ended: float | None = None
    lines: list[str] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    trace_path: str | None = None
    error: str | None = None
    stop_requested: bool = False
    proc: subprocess.Popen | None = field(default=None, repr=False)


RUNS: dict[str, RunState] = {}
RUNS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "kopt/2.0"

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/tasks":
            self.serve_tasks()
            return
        if path == "/api/loop/run":
            self.serve_run(query.get("path", [None])[0], _flag(query, "diff"))
            return
        if path == "/api/traces":
            self.serve_traces()
            return
        if path == "/api/report":
            self.serve_report()
            return
        if path == "/api/run":
            self.serve_run_state(query.get("id", [None])[0])
            return
        if path == "/api/runs":
            self.serve_runs()
            return
        self.serve_file(path)

    # ----------------------------------------------------------------- POST
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run", "/api/run/stop"}:
            self.respond(404, {"code": 404, "message": "接口不存在"})
            return
        payload = self._read_json()
        if payload is None:
            return
        if parsed.path == "/api/run/stop":
            self.stop_run(payload)
        else:
            self.start_run(payload)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self.respond(400, {"code": 400, "message": "请求体为空或过大"})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.respond(400, {"code": 400, "message": "请求体不是合法 JSON"})
            return None
        if not isinstance(payload, dict):
            self.respond(400, {"code": 400, "message": "请求体必须是 JSON 对象"})
            return None
        return payload

    # ---------------------------------------------------------------- 任务
    def serve_tasks(self) -> None:
        from catalogs import CATALOGS
        from proposers import LLMClient

        try:
            llm_available = LLMClient().available
        except Exception:  # noqa: BLE001 - 只是可用性提示
            llm_available = False
        items = []
        inbox = ROOT / "inbox"
        for spec in sorted(inbox.glob("*/task.json")):
            try:
                data = json.loads(spec.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            name = spec.parent.name
            bench = data.get("bench") or {}
            args = bench.get("args") or ([bench] if bench else [])
            shape = (args[0].get("shape") if args else None) or data.get("shape") or []
            items.append(
                {
                    "name": name,
                    "title": data.get("name") or name,
                    "entry": data.get("entry"),
                    "card": data.get("card", ""),
                    "target_ms": data.get("target_ms"),
                    "budget": int(data.get("budget", 8)),
                    "speedup_required": float(data.get("speedup_required", 1.1)),
                    "shape": shape,
                    "has_catalog": name in CATALOGS,
                }
            )
        self.respond(200, {"code": 0, "data": {"tasks": items, "llm_available": llm_available}})

    # ------------------------------------------------------------ 发起运行
    def start_run(self, payload: dict) -> None:
        task = str(payload.get("task") or "").strip()
        proposer = str(payload.get("proposer") or "scripted").strip()
        task_dir = _safe_task(task)
        if task_dir is None:
            self.respond(400, {"code": 400, "message": f"任务不存在：{task}"})
            return
        if proposer not in PROPOSERS:
            self.respond(400, {"code": 400, "message": f"proposer 必须是 {sorted(PROPOSERS)}"})
            return
        try:
            best_of = int(payload.get("best_of") or 1)
            budget = payload.get("budget")
            budget = int(budget) if budget not in (None, "") else None
        except (TypeError, ValueError):
            self.respond(400, {"code": 400, "message": "best_of / budget 必须是整数"})
            return
        if not 1 <= best_of <= 5:
            self.respond(400, {"code": 400, "message": "best_of 取值 1-5"})
            return
        if budget is not None and not 1 <= budget <= 50:
            self.respond(400, {"code": 400, "message": "budget 取值 1-50"})
            return

        with RUNS_LOCK:
            active = next((run for run in RUNS.values() if run.state in {"running", "stopping"}), None)
            if active is not None:
                self.respond(
                    409,
                    {"code": 409, "message": f"已有运行在进行：{active.id}", "data": _snapshot(active)},
                )
                return
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            WEB_RUN_DIR.mkdir(parents=True, exist_ok=True)
            out_path = WEB_RUN_DIR / f"{run_id}.json"
            cmd = [
                sys.executable,
                str(ROOT / "run.py"),
                str(task_dir),
                "--proposer",
                proposer,
                "--out",
                str(out_path),
                "--progress",
            ]
            if best_of > 1:
                cmd += ["--best-of", str(best_of)]
            if budget is not None:
                cmd += ["--budget", str(budget)]
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self.respond(500, {"code": 500, "message": f"启动失败：{exc}"})
                return
            state = RunState(id=run_id, task=task_dir.name, proposer=proposer, proc=proc)
            RUNS[run_id] = state
        threading.Thread(target=_pump, args=(state, out_path), daemon=True).start()
        self.respond(200, {"code": 0, "data": _snapshot(state)})

    def stop_run(self, payload: dict) -> None:
        run_id = str(payload.get("id") or "").strip()
        with RUNS_LOCK:
            state = RUNS.get(run_id) if run_id else _latest_run()
            if state is None:
                self.respond(404, {"code": 404, "message": "还没有运行记录"})
                return
            if state.state not in {"running", "stopping"}:
                self.respond(409, {"code": 409, "message": "这次运行已经结束", "data": _snapshot(state)})
                return
            if state.stop_requested:
                self.respond(200, {"code": 0, "data": _snapshot(state)})
                return
            state.stop_requested = True
            state.state = "stopping"
            snapshot = _snapshot(state)
            proc = state.proc
        if proc is not None:
            threading.Thread(target=_terminate, args=(proc,), daemon=True).start()
        self.respond(200, {"code": 0, "data": snapshot})

    def serve_run_state(self, run_id: str | None) -> None:
        with RUNS_LOCK:
            state = RUNS.get(run_id) if run_id else _latest_run()
            snapshot = _snapshot(state) if state else None
        if snapshot is None:
            self.respond(404, {"code": 404, "message": "还没有运行记录"})
            return
        self.respond(200, {"code": 0, "data": snapshot})

    def serve_runs(self) -> None:
        with RUNS_LOCK:
            items = [_snapshot(run) for run in sorted(RUNS.values(), key=lambda r: r.started, reverse=True)[:20]]
        self.respond(200, {"code": 0, "data": items})

    # ------------------------------------------------------------ 读轨迹
    def serve_run(self, raw: str | None, with_diff: bool = False) -> None:
        trace_path = _safe_trace(raw) if raw else _latest_trace()
        if trace_path is None or not trace_path.exists():
            self.respond(404, {"code": 404, "message": "还没有运行记录"})
            return
        try:
            data = json.loads(trace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.respond(500, {"code": 500, "message": f"轨迹不是合法 JSON：{exc}"})
            return
        data["_path"] = _relative(trace_path)
        if with_diff:
            _attach_diffs(data)
        self.respond(200, {"code": 0, "data": data})

    def serve_traces(self) -> None:
        items = []
        for path in sorted(TRACE_DIR.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if path.name.startswith("_"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not is_trace(data):
                continue
            metrics = data.get("metrics") or {}
            items.append(
                {
                    "path": _relative(path),
                    "task": data.get("task"),
                    "best": data.get("best"),
                    "speedup": metrics.get("speedup", data.get("speedup")),
                    "best_ms": metrics.get("best_ms", data.get("best_latency_ms")),
                    "target_met": metrics.get("target_met"),
                    "proposer": data.get("proposer"),
                    "rounds": len(data.get("rounds") or []),
                    "mtime": path.stat().st_mtime,
                }
            )
        self.respond(200, {"code": 0, "data": items})

    def serve_report(self) -> None:
        latest_by_task: dict[str, tuple[float, dict]] = {}
        for path in TRACE_DIR.glob("**/*.json"):
            if path.name.startswith("_"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not is_trace(data):
                continue
            task = data.get("task")
            mtime = path.stat().st_mtime
            if task not in latest_by_task or mtime > latest_by_task[task][0]:
                latest_by_task[task] = (mtime, data)
        traces = [item[1] for item in latest_by_task.values()]
        if not traces:
            self.respond(404, {"code": 404, "message": "还没有可汇总的轨迹"})
            return
        self.respond(200, {"code": 0, "data": summarize(traces)})

    # ------------------------------------------------------------ 静态文件
    def serve_file(self, path: str) -> None:
        rel = "web/index.html" if path in {"/", "/web", "/web/"} else path.lstrip("/")
        try:
            file = (ROOT / rel).resolve()
            file.relative_to(ROOT)
        except (ValueError, OSError):
            self.respond(404, {"code": 404, "message": "页面不存在"})
            return
        if not file.is_file():
            self.respond(404, {"code": 404, "message": "页面不存在"})
            return
        raw = file.read_bytes()
        kind = MIME.get(file.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def respond(self, status: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


# ---------------------------------------------------------------- 运行线程


def _pump(state: RunState, out_path: Path) -> None:
    """读子进程 stdout，解析 PROGRESS 行，结束后落运行状态。"""
    proc = state.proc
    assert proc is not None and proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("PROGRESS "):
                try:
                    event = json.loads(line[len("PROGRESS ") :])
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict):
                    with RUNS_LOCK:
                        state.rounds.append(event)
                        state.lines.append(_format_progress(event))
                        _trim(state)
                    continue
            if line:
                with RUNS_LOCK:
                    state.lines.append(line)
                    _trim(state)
    finally:
        code = proc.wait()
        with RUNS_LOCK:
            state.exit_code = code
            state.ended = time.time()
            if state.stop_requested:
                state.state = "cancelled"
                state.error = None
                state.lines.append("已手动停止这次运行。")
                _trim(state)
            else:
                state.state = "done" if code == 0 and out_path.exists() else "error"
                if code != 0:
                    state.error = f"run.py 退出码 {code}"
                elif not out_path.exists():
                    state.error = "运行结束但没有写出轨迹文件"
            if out_path.exists():
                state.trace_path = _relative(out_path)


def _terminate(proc: subprocess.Popen, grace_s: float = 3.0) -> None:
    """先 SIGTERM 整个进程组，宽限期内没退出再 SIGKILL。"""
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc, signal.SIGKILL)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _trim(state: RunState) -> None:
    if len(state.lines) > MAX_LINES:
        del state.lines[: len(state.lines) - MAX_LINES]


def _format_progress(event: dict) -> str:
    latency = event.get("latency_ms")
    latency_text = "-" if latency is None else f"{float(latency):.3f} ms"
    speedup = event.get("speedup_vs_baseline")
    speedup_text = "" if speedup is None else f"  {float(speedup):.2f}×"
    return (
        f"[第 {event.get('round')} 轮] {event.get('name')} · {event.get('status')}"
        f" · {latency_text}{speedup_text}  {event.get('message') or ''}"
    ).strip()


def _snapshot(state: RunState) -> dict:
    ended = state.ended or time.time()
    return {
        "id": state.id,
        "task": state.task,
        "proposer": state.proposer,
        "state": state.state,
        "exit_code": state.exit_code,
        "error": state.error,
        "started": state.started,
        "elapsed_s": ended - state.started,
        "lines": list(state.lines[-500:]),
        "rounds": list(state.rounds),
        "trace_path": state.trace_path,
        "stop_requested": state.stop_requested,
    }


def _latest_run() -> RunState | None:
    if not RUNS:
        return None
    return max(RUNS.values(), key=lambda run: run.started)


# ---------------------------------------------------------------- 工具函数


def _flag(query: dict, key: str) -> bool:
    return query.get(key, ["0"])[0] in {"1", "true", "yes"}


def _safe_task(name: str) -> Path | None:
    candidate = Path(name)
    if not name or candidate.name != name or name in {".", ".."}:
        return None
    task_dir = (ROOT / "inbox" / name).resolve()
    try:
        task_dir.relative_to((ROOT / "inbox").resolve())
    except ValueError:
        return None
    if not (task_dir / "task.json").is_file() or not (task_dir / "reference.py").is_file():
        return None
    return task_dir


def _attach_diffs(data: dict) -> None:
    rounds = data.get("rounds") or []
    baseline = rounds[0].get("source", "") if rounds else ""
    data["baseline_source"] = baseline
    for item in rounds:
        item["diff"] = _diff(baseline, item.get("source", ""))


def _diff(before: str, after: str, limit: int = 400) -> list[str]:
    if not before and not after:
        return []
    if before == after:
        return ["（与 reference.py 相同）"]
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="reference.py",
            tofile="candidate.py",
            lineterm="",
            n=3,
        )
    )
    if len(lines) > limit:
        lines = lines[:limit] + [f"… 省略 {len(lines) - limit} 行"]
    return lines


def _latest_trace() -> Path | None:
    """按修改时间取最近一次真实运行，包含页面发起的 traces/web/*.json。"""
    candidates = sorted(TRACE_DIR.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.name.startswith("_"):
            continue
        try:
            if is_trace(json.loads(path.read_text(encoding="utf-8"))):
                return path
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _safe_trace(raw: str) -> Path | None:
    try:
        path = (ROOT / raw).resolve()
        path.relative_to(ROOT)
    except (ValueError, OSError):
        return None
    return path if path.suffix == ".json" else None


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"http://127.0.0.1:{PORT}/web/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with RUNS_LOCK:
            active = [run.proc for run in RUNS.values() if run.state in {"running", "stopping"}]
        for proc in active:
            if proc is not None:
                _terminate(proc, grace_s=1.0)


if __name__ == "__main__":
    main()
