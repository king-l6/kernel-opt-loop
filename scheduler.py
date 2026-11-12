"""调度是普通程序：门禁顺序、预算、停止条件都在这里，生成器无权绕过。

Agent 每轮只拿到一份结构化上下文：
- 上一轮的门禁结果和人类可读反馈
- 当前最优源码、耗时、profile
- 已尝试过的候选摘要
它交回候选源码，不能直接写文件、不能改评测逻辑。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, is_dataclass

import devices
import memory as memory_mod
from feedback import compact_history, make_feedback
from policy import check_source
from probes import ProbeSession
from proposers import Patch, source_signature


def run_schedule(
    task,
    proposer,
    runner,
    config,
    env: dict | None = None,
    on_round=None,
    memory_store=None,
) -> dict:
    started = time.time()
    history: list[dict] = []
    tried: dict[str, int] = {}
    tried_names: list[str] = []
    best_name = "baseline"
    best_source = task.source
    best_ms = None
    best_profile: list = []
    baseline_ms = None
    experiments = 0
    evaluations = 0
    accepted = 0
    tokens = 0
    patience = 0
    stop = ""
    stop_reason = ""
    batch_id = 0
    empty_streak = 0

    baseline = runner.check("baseline", task.source, profile=True)
    if baseline.status != "ok":
        raise RuntimeError(f"baseline failed: {baseline.message}")
    baseline_ms = best_ms = baseline.latency_ms
    best_profile = baseline.profile or []
    # 设备描述跟着实际跑的那块卡走：要了 cuda 但只有 cpu 时，反馈里不能写显卡。
    spec = devices.spec_for(task, config)
    actual_device = getattr(baseline, "device", "") or ""
    if actual_device and actual_device != spec.backend:
        spec = devices.spec_for(task, config, actual_device)
    best_hardware = devices.analyze(spec, baseline_ms, getattr(baseline, "io_bytes", None))
    probes = ProbeSession(task, runner, config, on_event=on_round, spec=spec)
    memory_data = {}
    if memory_store is not None:
        try:
            memory_data = memory_store.load(task.name, spec) or {}
        except Exception:  # noqa: BLE001 - 记忆坏了不影响本次运行
            memory_data = {}
    tried[source_signature(task.source)] = 0
    feedback = make_feedback(
        status="baseline",
        message="accuracy passed",
        latency_ms=best_ms,
        best_ms=best_ms,
        baseline_ms=baseline_ms,
        target_ms=getattr(config, "target_ms", None),
        speedup_required=getattr(config, "speedup_required", 1.1),
        profile=best_profile,
        tried=[],
        history=[],
        round_no=0,
        hardware=best_hardware,
    )
    baseline_record = _record(
        round_no=0,
        name="baseline",
        title="原版",
        status="baseline",
        hypothesis="算法同学给的参考实现",
        expected_gain=None,
        source=task.source,
        eval_result=baseline,
        best_ms=best_ms,
        baseline_ms=baseline_ms,
        best_profile=best_profile,
        candidate_profile=best_profile,
        input_feedback=None,
        outcome_feedback=feedback,
        steps=_baseline_steps(task, baseline, best_profile, feedback),
        batch=None,
        batch_index=0,
        started=started,
        hardware=best_hardware,
    )
    history.append(baseline_record)
    _notify(on_round, baseline_record)

    if _under_target(best_ms, getattr(config, "target_ms", None)):
        stop, stop_reason = _hit_target(best_ms, config.target_ms), "target"
        trace = _trace(task, config, history, best_name, best_source, best_ms, baseline_ms, accepted, experiments, evaluations, tokens, stop, stop_reason, env, started, probes.used, spec)
        return _save_memory(trace, memory_store, task, spec)

    # 上次在这块卡上的最好版本先上机：通过就作为本轮的起点，
    # 省掉重复生成，也避免 Agent 再走一遍已知的弯路。
    warm_record = _warm_start(
        task,
        runner,
        memory_data,
        tried,
        config,
        spec,
        round_no=len(history),
        baseline_ms=baseline_ms,
        best_ms=best_ms,
        best_profile=best_profile,
        feedback=feedback,
        history=history,
    )
    if warm_record is not None:
        experiments += 1
        evaluations += 1
        history.append(warm_record)
        _notify(on_round, warm_record)
        feedback = warm_record["feedback_out"]
        if warm_record["status"] == "accepted":
            accepted += 1
            best_name = warm_record["name"]
            best_source = warm_record["source"]
            best_ms = warm_record["latency_ms"]
            best_profile = warm_record["profile"] or best_profile
            best_hardware = warm_record["hardware"] or best_hardware
            patience = 0
        if _under_target(best_ms, getattr(config, "target_ms", None)):
            stop, stop_reason = _hit_target(best_ms, config.target_ms), "target"
            trace = _trace(task, config, history, best_name, best_source, best_ms, baseline_ms, accepted, experiments, evaluations, tokens, stop, stop_reason, env, started, probes.used, spec)
            return _save_memory(trace, memory_store, task, spec)

    memory_view = memory_mod.summary(memory_data)
    while experiments < getattr(config, "budget", 8):
        if getattr(config, "max_tokens", None) and tokens >= int(config.max_tokens):
            stop, stop_reason = f"token 预算用完（{tokens}）。当前最好 {best_ms:.3f} ms。", "token_budget"
            break
        ctx = {
            "task_spec": task.prompt_spec(),
            "entry": task.entry,
            "reference_source": task.source,
            "current_best": {
                "name": best_name,
                "source": best_source,
                "latency_ms": best_ms,
                "profile": best_profile,
                "speedup_vs_baseline": _ratio(baseline_ms, best_ms),
            },
            "baseline_ms": baseline_ms,
            "target_ms": getattr(config, "target_ms", None),
            "feedback": feedback,
            "history": list(history),
            "round": len(history),
            "budget_left": max(0, int(getattr(config, "budget", 8)) - experiments),
            "best_of": int(getattr(config, "best_of", 1)),
            "device": str(getattr(config, "device", "cpu")),
            "device_info": spec.as_dict(),
            "hardware": best_hardware,
            "memory": memory_view,
            "triton_ready": bool(getattr(config, "triton_ready", False)),
        }
        probes.best_ms = best_ms
        probes.round_no = len(history)
        ctx["probe"] = probes.probe
        ctx["probe_budget"] = probes.left
        try:
            batch = proposer.propose(ctx, n=int(getattr(config, "best_of", 1)))
        except TypeError:
            batch = proposer.propose(ctx)  # 兼容只接收 ctx 的简单生成器
        except Exception as exc:  # noqa: BLE001 - 生成器故障不应丢掉已有轨迹
            stop, stop_reason = f"生成器出错：{type(exc).__name__}: {exc}", "proposer_error"
            break
        if not batch:
            reason = getattr(proposer, "last_error", "") or "没有给出下一版"
            kind = getattr(proposer, "last_error_kind", "")
            # 网关 5xx / 超时 / 响应格式错误是临时故障：连续两次才终止，
            # 不能因为一次网络抖动就放弃整个实验。模型主动放弃（declined）仍然立即停。
            if kind in {"transient", "parse"} and empty_streak < 2:
                empty_streak += 1
                continue
            stop = f"{reason}。当前最好是 {best_ms:.3f} ms。"
            stop_reason = "proposer_error" if kind in {"transient", "parse"} else "no_proposal"
            break
        empty_streak = 0

        batch_id += 1
        for index, raw in enumerate(batch):
            if experiments >= getattr(config, "budget", 8):
                break
            patch = _as_patch(raw)
            round_no = len(history)
            experiments += 1
            tokens += _tokens(patch.usage)
            tried_names.append(patch.name)
            round_started = time.perf_counter()

            status, message, error_kind, eval_result, candidate_profile = _gate(
                task, patch, runner, tried, round_no, config
            )
            if status == "evaluated":
                evaluations += 1
                status, message, error_kind, candidate_profile = _judge(
                    eval_result,
                    best_ms,
                    patch,
                    config,
                    runner,
                )
            latency = (
                eval_result.latency_ms
                if eval_result is not None and status in {"accepted", "rejected_slower"}
                else None
            )
            hardware = (
                devices.analyze(
                    spec,
                    eval_result.latency_ms,
                    getattr(eval_result, "io_bytes", None),
                )
                if eval_result is not None and eval_result.latency_ms is not None
                else None
            )
            if status == "accepted":
                accepted += 1
                best_name = patch.name
                best_source = patch.source or best_source
                best_ms = eval_result.latency_ms
                best_profile = candidate_profile or best_profile
                best_hardware = hardware or best_hardware
                patience = 0
            else:
                patience += 1
            outcome = make_feedback(
                status=status,
                message=message,
                error_kind=error_kind,
                latency_ms=latency,
                best_ms=best_ms,
                baseline_ms=baseline_ms,
                target_ms=getattr(config, "target_ms", None),
                speedup_required=getattr(config, "speedup_required", 1.1),
                profile=best_profile,
                candidate_profile=candidate_profile,
                tried=tried_names[-12:],
                history=compact_history(history, 6),
                round_no=round_no,
                hardware=hardware,
            )
            record = _record(
                round_no=round_no,
                name=patch.name,
                title=patch.title,
                status=status,
                hypothesis=patch.hypothesis,
                expected_gain=patch.expected_gain,
                source=patch.source or "",
                eval_result=eval_result,
                best_ms=best_ms,
                baseline_ms=baseline_ms,
                best_profile=best_profile,
                candidate_profile=candidate_profile,
                input_feedback=feedback,
                outcome_feedback=outcome,
                steps=_candidate_steps(patch, status, message, eval_result, candidate_profile, feedback, outcome),
                batch=batch_id,
                batch_index=index,
                started=round_started,
                tokens=patch.usage,
                agent=patch.meta,
                hardware=hardware,
            )
            history.append(record)
            _notify(on_round, record)
            feedback = outcome

            if _under_target(best_ms, getattr(config, "target_ms", None)):
                stop, stop_reason = _hit_target(best_ms, config.target_ms), "target"
                break
            if patience >= int(getattr(config, "patience", 4)):
                stop = f"连续 {patience} 轮没有超过当前最好 {best_ms:.3f} ms。"
                stop_reason = "patience"
                break
        if stop:
            break

    if not stop:
        stop = f"轮次用完。当前最好是 {best_ms:.3f} ms。"
        stop_reason = "budget"
    trace = _trace(
        task,
        config,
        history,
        best_name,
        best_source,
        best_ms,
        baseline_ms,
        accepted,
        experiments,
        evaluations,
        tokens,
        stop,
        stop_reason,
        env,
        started,
        probes.used,
        spec,
    )
    return _save_memory(trace, memory_store, task, spec)


# --------------------------------------------------------------------------
# 门禁与判定
# --------------------------------------------------------------------------


def _gate(task, patch: Patch, runner, tried, round_no, config):
    """返回 (status, message, error_kind, eval_result, candidate_profile)。

    status == "evaluated" 表示已经过沙箱，交给 _judge 判定快慢。
    """
    if patch.source is None:
        return "policy_error", "模型没有给出源码", "policy", None, []
    kind, message = check_source(patch.source, task.entry)
    if kind == "syntax":
        return "compile_error", message, "syntax", None, []
    if kind == "policy":
        return "policy_error", message, "policy", None, []
    signature = source_signature(patch.source)
    if signature in tried:
        return (
            "duplicate",
            f"与第 {tried[signature]} 轮的源码 AST 相同",
            "duplicate",
            None,
            [],
        )
    tried[signature] = round_no
    result = _check_quiet(runner, patch.name, patch.source, config)
    if result.status != "ok":
        return result.status, result.message, result.error_kind, result, []
    return "evaluated", result.message, result.error_kind, result, []


NOISE_CV = 0.15


def _check_quiet(runner, name: str, source: str, config):
    """噪声大时复测一次，取更稳的那次。

    单次 p50 在系统有后台负载时会失真（实测出现过 6× 偏差）。CV 超过阈值
    说明这次采样不可信：再跑一次，取更低的 p50；仍然不稳就按实测交给判定，
    并把 stats 一起留在轨迹里，方便事后归因。
    """
    result = runner.check(name, source, profile=False)
    cv = (getattr(result, "stats", None) or {}).get("cv")
    repeats = int(getattr(config, "repeats", 9) or 0)
    if result.status != "ok" or cv is None or cv <= NOISE_CV or repeats <= 1:
        return result
    again = runner.check(f"{name}-retry", source, profile=False)
    if again.status == "ok" and again.latency_ms and (
        not result.latency_ms or again.latency_ms < result.latency_ms
    ):
        return again
    return result


def _judge(result, best_ms, patch: Patch, config, runner):
    speedup = best_ms / result.latency_ms if result.latency_ms else 0.0
    required = float(getattr(config, "speedup_required", 1.1))
    if speedup >= required:
        profile = runner.profile(patch.name, patch.source) if hasattr(runner, "profile") else []
        return "accepted", f"耗时 {result.latency_ms:.3f} ms", "", profile
    return (
        "rejected_slower",
        f"耗时 {result.latency_ms:.3f} ms，没超过 {best_ms:.3f} ms",
        "",
        [],
    )


def _warm_start(
    task,
    runner,
    memory_data,
    tried,
    config,
    spec,
    *,
    round_no,
    baseline_ms,
    best_ms,
    best_profile,
    feedback,
    history,
):
    """把上次的最优源码先跑一遍；通过就走正常门禁，不通过也记一轮。"""
    source = memory_mod.best_source(memory_data)
    if not source or source == task.source:
        return None
    signature = source_signature(source)
    if signature in tried:
        return None
    tried[signature] = round_no

    result = _check_quiet(runner, "warm-start", source, config)
    required = float(getattr(config, "speedup_required", 1.1))
    latency = result.latency_ms
    if result.status != "ok":
        status, message, error_kind, profile = result.status, result.message, result.error_kind, []
    elif latency and best_ms / latency >= required:
        status, message, error_kind = "accepted", f"耗时 {latency:.3f} ms", ""
        profile = runner.profile("warm-start", source) if hasattr(runner, "profile") else []
    else:
        status = "rejected_slower"
        message = f"耗时 {latency:.3f} ms，没超过 {best_ms:.3f} ms" if latency else "没有测到耗时"
        error_kind, profile = "", []

    hypothesis = (memory_data.get("best") or {}).get("hypothesis") or "上一次在这块卡上的最优版本"
    hardware = devices.analyze(spec, latency, getattr(result, "io_bytes", None))
    outcome = make_feedback(
        status=status,
        message=message,
        error_kind=error_kind,
        latency_ms=latency if status in {"accepted", "rejected_slower"} else None,
        best_ms=best_ms,
        baseline_ms=baseline_ms,
        target_ms=getattr(config, "target_ms", None),
        speedup_required=required,
        profile=best_profile,
        candidate_profile=profile,
        tried=[],
        history=compact_history(history, 6),
        round_no=round_no,
        hardware=hardware,
    )
    patch = Patch(name="warm-start", title="上次最优", source=source, hypothesis=hypothesis)
    return _record(
        round_no=round_no,
        name="warm-start",
        title="上次最优",
        status=status,
        hypothesis=hypothesis,
        expected_gain=None,
        source=source,
        eval_result=result,
        best_ms=best_ms,
        baseline_ms=baseline_ms,
        best_profile=best_profile,
        candidate_profile=profile,
        input_feedback=feedback,
        outcome_feedback=outcome,
        steps=_candidate_steps(patch, status, message, result, profile, feedback, outcome),
        batch=None,
        batch_index=0,
        started=time.perf_counter(),
        hardware=hardware,
    )


def _save_memory(trace: dict, memory_store, task, spec) -> dict:
    if memory_store is None:
        return trace
    try:
        memory_store.update(task.name, spec, trace)
    except Exception:  # noqa: BLE001 - 记忆写失败不能影响轨迹
        pass
    return trace


def _notify(on_round, record: dict) -> None:
    """把一轮结果推给外部（页面实时进度）。回调失败不能影响调度。"""
    if on_round is None:
        return
    try:
        on_round(
            {
                "round": record["round"],
                "name": record["name"],
                "title": record["title"],
                "status": record["status"],
                "message": record["message"],
                "latency_ms": record["latency_ms"],
                "speedup_vs_baseline": record["speedup_vs_baseline"],
                "best_ms": record["best_ms"],
                "error_kind": record["error_kind"],
                "gbps": (record.get("hardware") or {}).get("gbps"),
            }
        )
    except Exception:  # noqa: BLE001 - 进度展示是旁路
        pass


def _as_patch(item) -> Patch:
    if isinstance(item, Patch):
        return item
    if isinstance(item, dict):
        return Patch(**item)
    raise TypeError(f"生成器必须返回 Patch 或 dict，收到 {type(item).__name__}")


def _tokens(usage: dict | None) -> int:
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if total is not None:
        return int(total)
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)


def _ratio(best, latency):
    if best is None or latency is None or latency <= 0:
        return None
    return best / latency


def _under_target(latency, target) -> bool:
    return target is not None and latency is not None and latency <= target


def _hit_target(latency, target) -> str:
    return f"耗时 {latency:.3f} ms，低于目标 {target:.3f} ms。"


# --------------------------------------------------------------------------
# 轨迹记录
# --------------------------------------------------------------------------


def _record(
    *,
    round_no,
    name,
    title,
    status,
    hypothesis,
    expected_gain,
    source,
    eval_result,
    best_ms,
    baseline_ms,
    best_profile,
    candidate_profile,
    input_feedback,
    outcome_feedback,
    steps,
    batch,
    batch_index,
    started,
    tokens=None,
    agent=None,
    hardware=None,
):
    latency = (
        eval_result.latency_ms
        if eval_result is not None and status in {"baseline", "accepted", "rejected_slower"}
        else None
    )
    return {
        "round": round_no,
        "name": name,
        "title": title,
        "status": status,
        "hypothesis": hypothesis,
        "expected_gain": expected_gain,
        "message": (eval_result.message if eval_result is not None else outcome_feedback["message"]),
        "max_abs_err": eval_result.max_abs_err if eval_result is not None else None,
        "max_rel_err": eval_result.max_rel_err if eval_result is not None else None,
        "latency_ms": latency,
        "stats": eval_result.stats if eval_result is not None else None,
        "speedup_vs_baseline": _ratio(baseline_ms, latency),
        "speedup_vs_best": _ratio(best_ms, latency) if status == "rejected_slower" else None,
        "source": source,
        "profile": candidate_profile or [],
        "best_profile": best_profile or [],
        "best_ms": best_ms,
        "error_kind": getattr(eval_result, "error_kind", "") if eval_result is not None else outcome_feedback.get("error_kind", ""),
        "device": getattr(eval_result, "device", "") if eval_result is not None else "",
        "threads": getattr(eval_result, "threads", None) if eval_result is not None else None,
        "hardware": hardware or {},
        "clocks": getattr(eval_result, "clocks", None) if eval_result is not None else None,
        "wall_s": getattr(eval_result, "wall_s", 0.0) if eval_result is not None else 0.0,
        "tokens": tokens or {},
        "agent": agent or {},
        "batch": batch,
        "batch_index": batch_index,
        "feedback_in": input_feedback,
        "feedback_out": outcome_feedback,
        "steps": steps,
        "elapsed_s": time.perf_counter() - started,
    }


def _baseline_steps(task, result, profile, feedback):
    return [
        {
            "state": "pass",
            "title": "读入算法同学的文件",
            "detail": f"{task.reference_file} 里的 {task.entry}。这份输出是对照，这份耗时是基线。",
        },
        {
            "state": "pass",
            "title": "上机跑原版（沙箱子进程）",
            "detail": _ran(result),
        },
        {
            "state": "info",
            "title": "看耗时落在哪",
            "detail": _profile_detail(profile),
            "bars": _bars(profile),
        },
        {"state": "info", "title": "交给改写的一句话", "detail": feedback["text"]},
    ]


def _candidate_steps(patch, status, message, result, profile, input_feedback, outcome):
    steps = [
        {
            "state": "info",
            "title": "改写收到的上下文",
            "detail": (input_feedback or {}).get("text", "第一轮"),
        },
        {
            "state": "info",
            "title": "这一版的假设",
            "detail": patch.hypothesis or patch.title,
        },
        {"state": "info", "title": "交回一版", "detail": patch.title},
    ]
    if status == "policy_error":
        steps += [
            {"state": "fail", "title": "策略检查", "detail": message},
            {"state": "skip", "title": "上机", "detail": "源码没有交给机器。"},
        ]
    elif status == "duplicate":
        steps += [
            {"state": "fail", "title": "查重", "detail": message},
            {"state": "skip", "title": "上机", "detail": "同一份 AST 不再重复消耗实验。"},
        ]
    elif status == "compile_error":
        steps += [
            {"state": "fail", "title": "编译", "detail": message},
            {"state": "skip", "title": "对结果与测耗时", "detail": "没有加载成功。"},
        ]
    elif status == "accuracy_fail":
        steps += [
            {"state": "pass", "title": "编译", "detail": "通过。"},
            {"state": "fail", "title": "对结果", "detail": message},
            {"state": "skip", "title": "测耗时", "detail": "结果没过，不测耗时。"},
        ]
    elif status in {"timeout", "crash"}:
        steps += [
            {"state": "fail", "title": "执行", "detail": message},
            {"state": "skip", "title": "对结果与测耗时", "detail": "子进程没有给出结果。"},
        ]
    elif status == "rejected_slower":
        steps += [
            {"state": "pass", "title": "编译", "detail": "通过。"},
            {"state": "pass", "title": "对结果", "detail": "和参考实现一致。"},
            {"state": "fail", "title": "测耗时", "detail": message},
        ]
    elif status == "accepted":
        steps += [
            {"state": "pass", "title": "编译", "detail": "通过。"},
            {"state": "pass", "title": "对结果", "detail": "和参考实现一致。"},
            {"state": "pass", "title": "测耗时", "detail": message + "，比当前最好快，留下这一版。"},
            {
                "state": "pass",
                "title": "看耗时落在哪",
                "detail": _profile_detail(profile),
                "bars": _bars(profile),
            },
        ]
    steps.append({"state": "info", "title": "交给下一版的一句", "detail": outcome["text"]})
    return steps


def _ran(result) -> str:
    detail = f"同一批输入跑完，耗时 {result.latency_ms:.3f} ms。"
    if result.stats:
        detail += f" p90 {result.stats['p90_ms']:.3f} ms，CV {result.stats['cv']:.1%}。" if result.stats.get("cv") is not None else ""
    return detail


def _profile_detail(profile) -> str:
    if not profile:
        return "这次没有采到分项。"
    top = profile[0]
    return f"最耗时的是 {top['name']}，{top['us']:.1f} μs（{top.get('pct', 0):.0f}%）。"


def _bars(profile):
    if not profile:
        return []
    return [{"name": item["name"], "value": item["us"]} for item in profile]


# --------------------------------------------------------------------------
# 轨迹
# --------------------------------------------------------------------------


def _trace(
    task,
    config,
    history,
    best_name,
    best_source,
    best_ms,
    baseline_ms,
    accepted,
    experiments,
    evaluations,
    tokens,
    stop,
    stop_reason,
    env,
    started,
    probes=0,
    spec=None,
) -> dict:
    speedup = _ratio(baseline_ms, best_ms)
    return {
        "schema_version": 2,
        "task": task.name,
        "task_dir": str(task.root),
        "reference_file": task.reference_file,
        "entry": task.entry,
        "card": task.card,
        "shape": task.shape,
        "target_ms": getattr(config, "target_ms", None),
        "device": str(getattr(config, "device", "cpu")),
        "device_info": spec.as_dict() if spec is not None else {},
        "config": _config_dict(config),
        "env": env or {},
        "proposer": getattr(config, "proposer_label", None),
        "best": best_name,
        "best_source": best_source,
        "best_latency_ms": best_ms,
        "baseline_ms": baseline_ms,
        "speedup": speedup,
        "accepted": accepted,
        "experiments": experiments,
        "evaluations": evaluations,
        "probes": probes,
        "tokens": tokens,
        "stop": stop,
        "stop_reason": stop_reason,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_s": time.time() - started,
        "metrics": {
            "baseline_ms": baseline_ms,
            "best_ms": best_ms,
            "speedup": speedup,
            "target_ms": getattr(config, "target_ms", None),
            "target_met": (
                _under_target(best_ms, getattr(config, "target_ms", None))
                if getattr(config, "target_ms", None) is not None
                else None
            ),
            "accepted": accepted,
            "experiments": experiments,
            "evaluations": evaluations,
            "probes": probes,
            "tokens": tokens,
            "rounds": len(history) - 1,
            "stop_reason": stop_reason,
        },
        "rounds": history,
    }


def _config_dict(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}
