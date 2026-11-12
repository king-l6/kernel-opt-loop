"""多设备矩阵：设备不可用要跳过、汇总要按 (算子, 设备) 对齐。"""

import json
from pathlib import Path

from fleet import _slug, matrix_text, run_fleet, summarize


def test_slug_is_filename_safe():
    assert _slug("cuda:0") == "cuda-0"
    assert _slug("cpu") == "cpu"


def test_summarize_and_matrix_align_rows():
    rows = [
        {
            "task": "softmax",
            "device": "cpu",
            "status": "ok",
            "speedup": 2.0,
            "target_met": True,
            "tokens": 10,
            "experiments": 2,
            "probes": 1,
            "elapsed_s": 1.0,
        },
        {
            "task": "softmax",
            "device": "npu:0",
            "status": "skipped",
            "error": "设备不可用",
            "elapsed_s": 0.0,
        },
    ]
    summary = summarize(rows, [Path("inbox/softmax")], ["cpu", "npu:0"])
    assert summary["totals"]["ok"] == 1
    assert summary["totals"]["skipped"] == 1
    assert summary["totals"]["successes"] == 1
    text = matrix_text(summary)
    assert "softmax" in text
    assert "2.00x" in text
    assert "跳过" in text
    assert "不可用" in text


def test_unavailable_device_is_skipped_without_running(tmp_path):
    summary = run_fleet(
        [Path("inbox/softmax")],
        ["npu:0"],
        budget=1,
        out_dir=tmp_path,
    )
    assert summary["totals"]["runs"] == 1
    assert summary["totals"]["skipped"] == 1
    assert summary["rows"][0]["status"] == "skipped"
    saved = json.loads((tmp_path / "_summary.json").read_text(encoding="utf-8"))
    assert saved["devices"][0]["available"] is False
