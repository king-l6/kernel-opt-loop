"""控制台的本地边界：任务名不能穿越目录，停止必须真的结束进程。"""

import signal
import subprocess
import sys

import serve


def test_safe_task_accepts_inbox_tasks():
    task = serve._safe_task("softmax")
    assert task is not None
    assert task.name == "softmax"


def test_safe_task_rejects_traversal_and_missing_tasks():
    for name in ["", ".", "..", "../inbox", "softmax/../../etc", "/etc/passwd", "not_a_task"]:
        assert serve._safe_task(name) is None


def test_safe_trace_stays_inside_project():
    assert serve._safe_trace("traces/latest.json") is not None
    assert serve._safe_trace("../outside.json") is None
    assert serve._safe_trace("traces/latest.txt") is None


def test_terminate_stops_a_running_process():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        serve._terminate(proc, grace_s=2.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            serve._signal_group(proc, signal.SIGKILL)
            proc.wait()
