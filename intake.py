"""从一个目录读入算法同学交给的任务。

task.json 只描述“怎么验、怎么测”，reference.py 只描述“标准答案”。
父进程不执行 reference.py；它会被送进沙箱子进程。
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Task:
    root: Path
    name: str
    entry: str
    card: str
    source: str
    accuracy: list[dict]
    bench: dict
    target_ms: float | None = None
    budget: int = 8
    patience: int = 4
    repeats: int = 9
    warmup: int = 3
    speedup_required: float = 1.1
    timeout_s: float = 30.0
    memory_mb: int = 0
    threads: int | None = None
    profile_limit: int = 5
    dtype: str = "float32"
    seed: int = 0
    temperature: float = 0.7
    extras: dict = field(default_factory=dict)

    @property
    def reference_file(self) -> str:
        return str(self.root / "reference.py")

    @property
    def shape(self) -> list[int]:
        """兼容旧讲解页：返回基准输入的第一个张量形状。"""
        args = self.bench.get("args") or [self.bench]
        first = args[0] if args else {}
        return list(first.get("shape", []))

    def prompt_spec(self) -> dict:
        return {
            "name": self.name,
            "entry": self.entry,
            "dtype": self.dtype,
            "accuracy_shapes": _shapes(self.accuracy),
            "bench_shapes": _shapes([self.bench]),
            "target_ms": self.target_ms,
            "speedup_required": self.speedup_required,
        }

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("source", None)
        data["root"] = str(self.root)
        return data


def load_task(path: str | Path) -> Task:
    root = Path(path)
    spec_path = root / "task.json"
    ref_path = root / "reference.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} 不存在")
    if not ref_path.exists():
        raise FileNotFoundError(f"{ref_path} 不存在")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source = ref_path.read_text(encoding="utf-8")
    entry = spec.get("entry")
    if not entry:
        raise RuntimeError(f"{spec_path} 缺少 entry")
    _ensure_entry(source, entry, ref_path)

    dtype = spec.get("dtype", "float32")
    accuracy = spec.get("accuracy")
    if not accuracy:
        accuracy = _legacy_accuracy(spec.get("shape"), dtype)
    bench = spec.get("bench")
    if not bench:
        shape = spec.get("shape")
        if not shape:
            raise RuntimeError(f"{spec_path} 需要 bench 或 shape")
        bench = {"args": [{"shape": list(shape), "dist": "randn", "seed": 1, "dtype": dtype}]}

    return Task(
        root=root,
        name=spec.get("name") or root.name,
        entry=entry,
        card=spec.get("card", "本机 CPU"),
        source=source,
        accuracy=[_normalize_case(case, dtype) for case in accuracy],
        bench=_normalize_case(bench, dtype),
        target_ms=_optional_float(spec.get("target_ms")),
        budget=int(spec.get("budget", 8)),
        patience=int(spec.get("patience", 4)),
        repeats=int(spec.get("repeats", 9)),
        warmup=int(spec.get("warmup", 3)),
        speedup_required=float(spec.get("speedup_required", 1.1)),
        timeout_s=float(spec.get("timeout_s", 30.0)),
        memory_mb=int(spec.get("memory_mb", 0)),
        threads=_optional_int(spec.get("threads")),
        profile_limit=int(spec.get("profile_limit", 5)),
        dtype=dtype,
        seed=int(spec.get("seed", 0)),
        temperature=float(spec.get("temperature", 0.7)),
        extras={key: value for key, value in spec.items() if key not in _KNOWN_KEYS},
    )


def list_task_dirs(root: str | Path) -> list[Path]:
    base = Path(root)
    if (base / "task.json").exists():
        return [base]
    return sorted(p.parent for p in base.glob("*/task.json"))


def _ensure_entry(source: str, entry: str, path: Path) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuntimeError(f"{path} 语法错误：{exc.msg} (line {exc.lineno})") from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == entry:
                    return
    raise RuntimeError(f"{path} 里没有函数 {entry}")


def _normalize_case(case: dict, dtype: str) -> dict:
    if not isinstance(case, dict):
        raise RuntimeError(f"输入描述必须是对象，收到 {type(case).__name__}")
    case = dict(case)
    if "args" in case:
        case["args"] = [dict(arg, dtype=arg.get("dtype", case.get("dtype", dtype))) for arg in case["args"]]
    else:
        case["dtype"] = case.get("dtype", dtype)
    return case


def _legacy_accuracy(shape, dtype: str) -> list[dict]:
    """旧格式（只有 shape）时使用的固定对照集，额外补了边界输入。"""
    return [
        {"args": [{"shape": [8, 32], "dist": "randn", "seed": 0}]},
        {"args": [{"shape": [4, 16, 32], "dist": "randn", "seed": 0}]},
        {"args": [{"shape": [1, 3], "dist": "constant", "value": 1000.0}]},
        {"args": [{"shape": [2, 4], "dist": "zeros"}]},
        {"args": [{"shape": [4, 8], "dist": "randn", "seed": 2, "transpose": True}]},
    ]


def _shapes(cases: list[dict]) -> list:
    shapes = []
    for case in cases:
        args = case.get("args") or [case]
        shapes.append([list(arg.get("shape", [])) for arg in args])
    return shapes


def _optional_float(value):
    return None if value is None else float(value)


def _optional_int(value):
    return None if value is None else int(value)


_KNOWN_KEYS = {
    "name",
    "entry",
    "card",
    "shape",
    "dtype",
    "accuracy",
    "bench",
    "target_ms",
    "budget",
    "patience",
    "repeats",
    "warmup",
    "speedup_required",
    "timeout_s",
    "memory_mb",
    "threads",
    "profile_limit",
    "seed",
    "temperature",
}
