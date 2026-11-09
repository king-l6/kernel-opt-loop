"""把评测结果压成两样东西：一份结构化反馈给 Agent，一句人话给轨迹页。"""

from __future__ import annotations


def make_feedback(
    *,
    status: str,
    message: str,
    error_kind: str = "",
    latency_ms: float | None = None,
    best_ms: float | None = None,
    baseline_ms: float | None = None,
    target_ms: float | None = None,
    speedup_required: float = 1.1,
    profile: list | None = None,
    candidate_profile: list | None = None,
    tried: list | None = None,
    history: list | None = None,
    round_no: int = 0,
) -> dict:
    profile = profile or []
    tried = tried or []
    history = history or []
    speedup_best = _ratio(best_ms, latency_ms)
    speedup_base = _ratio(baseline_ms, latency_ms)
    hints = _hints(status, error_kind, message, speedup_best, speedup_required, profile, target_ms)
    feedback = {
        "round": round_no,
        "status": status,
        "error_kind": error_kind,
        "message": message,
        "latency_ms": latency_ms,
        "best_ms": best_ms,
        "baseline_ms": baseline_ms,
        "target_ms": target_ms,
        "speedup_vs_best": speedup_best,
        "speedup_vs_baseline": speedup_base,
        "speedup_required": speedup_required,
        "profile": profile,
        "candidate_profile": candidate_profile or [],
        "tried": tried,
        "history": history,
        "hints": hints,
    }
    feedback["text"] = _text(feedback)
    return feedback


def compact_history(records: list[dict], limit: int = 6) -> list[dict]:
    out = []
    for item in records[-limit:]:
        out.append(
            {
                "round": item.get("round"),
                "name": item.get("name"),
                "status": item.get("status"),
                "hypothesis": item.get("hypothesis") or "",
                "latency_ms": item.get("latency_ms"),
                "message": (item.get("message") or "")[:240],
            }
        )
    return out


def _text(feedback: dict) -> str:
    status = feedback["status"]
    latency = feedback["latency_ms"]
    best = feedback["best_ms"]
    baseline = feedback["baseline_ms"]
    target = feedback["target_ms"]
    message = feedback["message"]
    if status == "baseline":
        text = f"基线耗时 {_ms(latency)}。"
        if target is not None:
            text += f"目标 {_ms(target)}。"
        if feedback["profile"]:
            top = feedback["profile"][0]
            text += f"最耗时的是 {top['name']}（{top.get('pct', 0):.0f}%）。"
        return text
    if status == "compile_error":
        return f"编译没过：{message}。先修到能加载。"
    if status == "policy_error":
        return f"策略拒绝：{message}。换一种写法。"
    if status == "accuracy_fail":
        return f"结果不一致：{message}。先修正确性，暂不测速。"
    if status == "timeout":
        return f"{message}。检查死循环，或者把逐行 Python 循环改成张量运算。"
    if status == "crash":
        return f"执行崩溃：{message}。先让它稳定返回。"
    if status == "duplicate":
        return "这一版源码之前已经试过，换一个真正不同的方向。"
    if status == "accepted":
        text = f"耗时 {_ms(latency)}，已留下，比基线快 {_x(feedback['speedup_vs_baseline'])}。"
        if target is not None and latency is not None and latency <= target:
            text += f"低于目标 {_ms(target)}。"
        if feedback["candidate_profile"]:
            top = feedback["candidate_profile"][0]
            text += f"现在最耗时的是 {top['name']}（{top.get('pct', 0):.0f}%）。"
        return text
    if status == "rejected_slower":
        return (
            f"结果一致，耗时 {_ms(latency)}，没超过当前最好 {_ms(best)}"
            f"（需要快 {feedback['speedup_required']:.1f}×）。"
        )
    return message


def _hints(status, error_kind, message, speedup_best, required, profile, target) -> list[str]:
    hints: list[str] = []
    if status == "compile_error":
        if error_kind == "syntax":
            hints.append("修语法：括号、缩进、函数体。")
        else:
            hints.append("修加载错误：函数名、导入、全局变量。")
    elif status == "policy_error":
        hints.append("只允许 torch / triton / math / typing / numpy，不要碰文件、网络、进程。")
    elif status == "accuracy_fail":
        if error_kind == "non_finite":
            hints.append("先做数值稳定化：softmax/log_softmax 减最大值，norm 加 eps。")
        elif error_kind == "shape":
            hints.append("保持输出形状与输入一致，检查 reduction 的 dim。")
        elif error_kind == "dtype":
            hints.append("输出 dtype 必须与输入一致。")
        else:
            hints.append(f"检查 reduction 维度和归一化分母：{message}")
    elif status == "timeout":
        hints.append("去掉逐行 Python 循环，改成整块张量运算；检查是否有无界循环。")
    elif status == "rejected_slower":
        hints.append("减少中间张量，融合 exp/sum/div，避免额外的 copy 和 Python 循环。")
    elif status == "duplicate":
        hints.append("不要重复提交同一份 AST；换优化手段（向量化、融合、编译、Triton）。")
    elif status == "accepted":
        hints.append("已采纳；继续压当前 profile 里占比最高的算子，或者换更大的优化手段。")
    if speedup_best is not None and speedup_best < required and status == "rejected_slower":
        hints.append(f"还差 {(required / speedup_best - 1) * 100:.0f}% 才到采纳线。")
    if profile:
        top = profile[0]
        hints.append(f"当前最优的瓶颈是 {top['name']}（{top.get('pct', 0):.0f}%）。")
    return hints


def _ratio(best, latency):
    if best is None or latency is None or latency <= 0:
        return None
    return best / latency


def _ms(value) -> str:
    if value is None:
        return "未测"
    return f"{value:.3f} ms"


def _x(value) -> str:
    if value is None:
        return "未知"
    return f"{value:.2f}×"
