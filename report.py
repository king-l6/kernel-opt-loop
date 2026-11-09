"""汇总 trajectories：成功率、几何平均加速、单位实验收益。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import is_trace, report_text, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总轨迹指标")
    parser.add_argument("paths", nargs="*", default=["traces"], help="轨迹文件或目录")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--out", help="把汇总写到文件")
    args = parser.parse_args()

    files = _expand(args.paths)
    traces = []
    skipped = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            skipped.append((path, str(exc)))
            continue
        if not is_trace(data):
            skipped.append((path, "不是本流水线的轨迹"))
            continue
        traces.append(data)
    if not traces:
        raise SystemExit("没有找到可汇总的轨迹")
    summary = summarize(traces)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(report_text(summary))
        if skipped:
            print(f"\n跳过 {len(skipped)} 个文件：")
            for path, reason in skipped:
                print(f"  {path}: {reason}")


def _expand(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.json") if not p.name.startswith("_")))
        elif path.exists():
            files.append(path)
        else:
            print(f"跳过不存在的路径：{path}")
    return files


if __name__ == "__main__":
    main()
