"""设备反馈：设备识别、带宽下界、以及子进程回报的最小 I/O。"""

from types import SimpleNamespace

import devices
from feedback import make_feedback
from gates import Runner
from intake import load_task
from loop import Config

TASK = load_task("inbox/softmax")


def _config(**overrides):
    config = Config.from_task(TASK)
    config.repeats = 1
    config.warmup = 0
    config.threads = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_registry_matches_card_name():
    spec = devices.detect("cpu", {"name": "NVIDIA A100-SXM4-80GB"})
    assert spec.peak_bw_gbps == 2039.0
    assert spec.source == "spec"


def test_override_peak_wins():
    spec = devices.detect("cpu", {"peak_bw_gbps": 42.0, "name": "自研卡"})
    assert spec.peak_bw_gbps == 42.0
    assert spec.source == "override"
    assert spec.name == "自研卡"


def test_unknown_card_reports_bandwidth_without_percentage():
    spec = devices.detect("cpu", {"name": "某块没登记的卡"})
    assert spec.peak_bw_gbps is None
    hardware = devices.analyze(spec, latency_ms=2.0, io_bytes=100_000_000)
    assert hardware["gbps"] == 50.0
    assert hardware["bw_pct"] is None
    assert "峰值" in hardware["advice"][0]


def test_analyze_computes_floor_and_headroom():
    spec = devices.DeviceSpec(key="cpu", backend="cpu", name="测试卡", peak_bw_gbps=100.0)
    hardware = devices.analyze(spec, latency_ms=2.0, io_bytes=100_000_000)
    assert hardware["gbps"] == 50.0
    assert hardware["bw_pct"] == 0.5
    assert abs(hardware["floor_ms"] - 1.0) < 1e-9
    assert hardware["io_headroom"] == 2.0
    assert hardware["cache_resident"] is False


def test_analyze_flags_cache_resident_small_working_set():
    spec = devices.DeviceSpec(key="cpu", backend="cpu", name="测试卡", peak_bw_gbps=100.0)
    hardware = devices.analyze(spec, latency_ms=0.001, io_bytes=1_000_000)
    assert hardware["bw_pct"] > 1.0
    assert hardware["cache_resident"] is True
    assert "缓存" in hardware["advice"][0]


def test_analyze_without_measurement_is_empty():
    spec = devices.DeviceSpec(key="cpu", backend="cpu", name="测试卡", peak_bw_gbps=100.0)
    hardware = devices.analyze(spec, None, None)
    assert hardware["gbps"] is None
    assert hardware["advice"] == []


def test_spec_for_uses_cli_peak_over_task_spec():
    task = SimpleNamespace(device_spec={"peak_bw_gbps": 10.0, "name": "任务声明卡"})
    config = SimpleNamespace(device="cpu", peak_bw_gbps=99.0)
    spec = devices.spec_for(task, config)
    assert spec.peak_bw_gbps == 99.0
    assert spec.source == "override"


def test_worker_reports_min_io_bytes():
    result = Runner(TASK, _config()).check("baseline", TASK.source)
    assert result.status == "ok"
    # 512x1024 float32 输入 + 同样大小的输出
    assert result.io_bytes == 512 * 1024 * 4 * 2


def test_feedback_carries_hardware_advice():
    hardware = devices.analyze(
        devices.DeviceSpec(key="cpu", backend="cpu", name="测试卡", peak_bw_gbps=100.0),
        latency_ms=2.0,
        io_bytes=100_000_000,
    )
    feedback = make_feedback(
        status="accepted",
        message="耗时 2.000 ms",
        latency_ms=2.0,
        best_ms=2.0,
        baseline_ms=10.0,
        hardware=hardware,
    )
    assert feedback["hardware"]["gbps"] == 50.0
    assert "有效带宽" in feedback["text"]
    assert any("纸面带宽" in hint for hint in feedback["hints"])


def test_trace_records_device_and_hardware():
    from scheduler import run_schedule

    class OneShot:
        kind = "scripted"

        def __init__(self):
            self.done = False

        def propose(self, ctx, n=1):
            from proposers import Patch

            if self.done:
                return []
            self.done = True
            return [
                Patch(
                    name="vec",
                    title="整块",
                    source=(
                        "def softmax(x):\n"
                        "    m = x.max(dim=-1, keepdim=True).values\n"
                        "    e = torch.exp(x - m)\n"
                        "    return e / e.sum(dim=-1, keepdim=True)\n"
                    ),
                    hypothesis="向量化",
                )
            ]

    config = _config(probe_budget=0)
    trace = run_schedule(TASK, OneShot(), Runner(TASK, config), config)
    assert trace["device_info"]["backend"] == "cpu"
    assert trace["device_info"]["name"]
    baseline = trace["rounds"][0]
    assert baseline["hardware"]["io_bytes"] == 512 * 1024 * 4 * 2
    assert baseline["hardware"]["gbps"] > 0
    assert trace["rounds"][-1]["hardware"]["device"] == trace["device_info"]["name"]
