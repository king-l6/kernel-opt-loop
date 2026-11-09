"""讲解页与只读 API。

- GET /api/loop/run?path=traces/xxx.json  读一条轨迹
- GET /api/traces                          列出可切换的轨迹
- GET /api/report                          汇总 traces/ 下的全部轨迹
- GET /web/                                讲解页
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from metrics import is_trace, summarize

ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "traces"
PORT = 8765
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/loop/run":
            self.serve_run(query.get("path", [None])[0])
            return
        if path == "/api/traces":
            self.serve_traces()
            return
        if path == "/api/report":
            self.serve_report()
            return
        self.serve_file(path)

    def serve_run(self, raw: str | None) -> None:
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
                    "mtime": path.stat().st_mtime,
                }
            )
        self.respond(200, {"code": 0, "data": items})

    def serve_report(self) -> None:
        traces = []
        for path in sorted(TRACE_DIR.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if is_trace(data):
                traces.append(data)
        if not traces:
            self.respond(404, {"code": 404, "message": "还没有可汇总的轨迹"})
            return
        self.respond(200, {"code": 0, "data": summarize(traces)})

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
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _latest_trace() -> Path | None:
    latest = TRACE_DIR / "latest.json"
    if latest.exists():
        return latest
    candidates = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


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
    server.serve_forever()


if __name__ == "__main__":
    main()
