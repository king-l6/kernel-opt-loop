"""候选源码的静态策略检查。

沙箱负责“跑不坏宿主”，这里负责“别把明显越界的东西送进沙箱”。
检查是保守的白名单，不追求完备；真正的安全边界仍然是进程/容器。
"""

from __future__ import annotations

import ast

MAX_SOURCE_CHARS = 30_000
ALLOWED_IMPORTS = {"torch", "triton", "math", "typing", "numpy", "__future__"}
MODULE_ROOTS = {"torch", "np", "numpy", "math", "triton", "tl"}
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "setattr",
    "delattr",
}
FORBIDDEN_ATTR_CALLS = {"system", "popen", "spawn", "fork", "__setattr__", "__delattr__", "__setitem__"}
FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "ctypes",
    "pickle",
    "importlib",
    "builtins",
    "resource",
    "signal",
    "threading",
    "multiprocessing",
}


def check_source(source: str, entry: str) -> tuple[str, str]:
    """返回 ("", "") 表示通过，否则返回 (kind, message)，kind 为 syntax/policy。"""
    if not isinstance(source, str) or not source.strip():
        return "policy", "源码为空"
    if len(source) > MAX_SOURCE_CHARS:
        return "policy", f"源码超过 {MAX_SOURCE_CHARS} 字符"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return "syntax", f"SyntaxError: {exc.msg} (line {exc.lineno})"

    defines_entry = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            defines_entry = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_MODULES:
                    return "policy", f"禁止 import {alias.name}"
                if root not in ALLOWED_IMPORTS:
                    return "policy", f"只允许 import {sorted(ALLOWED_IMPORTS)}，收到 {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_MODULES:
                return "policy", f"禁止 from {node.module} import"
            if root and root not in ALLOWED_IMPORTS:
                return "policy", f"只允许 import {sorted(ALLOWED_IMPORTS)}，收到 {node.module}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_mutates_module(target) for target in targets):
                return "policy", "禁止改写模块或类的属性"
        elif isinstance(node, ast.Delete):
            if any(_mutates_module(target) for target in node.targets):
                return "policy", "禁止删除模块或类的属性"
        elif isinstance(node, ast.Attribute) and node.attr in {"__dict__", "__class__"}:
            return "policy", f"禁止访问 {node.attr}"
        elif isinstance(node, ast.Call):
            # 只拦裸调用（eval/exec/compile/open...）。
            # torch.compile、model.eval() 这类属性调用是合法的。
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return "policy", f"禁止调用 {node.func.id}()"
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTR_CALLS:
                return "policy", f"禁止调用 {node.func.attr}()"
    if not defines_entry:
        return "policy", f"必须定义函数 {entry}(...)"
    return "", ""


def _root_name(node) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _mutates_module(target) -> bool:
    """赋值/删除的目标是否指向 torch、numpy 等模块或它们的类属性。

    候选代码改写 torch.allclose / torch.Tensor.abs 就能骗过精度门，
    所以这类写入在进沙箱前就拒绝。
    """
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_mutates_module(item) for item in target.elts)
    if isinstance(target, ast.Attribute):
        return _root_name(target) in MODULE_ROOTS
    if isinstance(target, ast.Subscript):
        value = target.value
        if isinstance(value, ast.Attribute) and value.attr == "__dict__":
            return True
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "vars":
            return True
    return False
