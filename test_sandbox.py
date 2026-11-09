import pytest

from gates import Runner
from intake import load_task
from loop import Config

TASK = load_task("inbox/softmax")

STABLE = """
def softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)
"""

WRONG_AXIS = """
def softmax(x):
    m = x.max(dim=0, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=0, keepdim=True)
"""

BROKEN = """
def softmax(x):
    return torch.exp(x
"""

HANG = """
def softmax(x):
    while True:
        pass
"""

EXIT = """
def softmax(x):
    import os
    os._exit(3)
"""


def _config(**overrides):
    values = dict(repeats=1, warmup=0, threads=1, timeout_s=20.0)
    values.update(overrides)
    return Config(**values)


def test_timeout_kills_the_child():
    runner = Runner(TASK, _config(timeout_s=2.0))
    result = runner.check("hang", HANG)
    assert result.status == "timeout"
    assert result.latency_ms is None


def test_compile_error_has_no_latency():
    runner = Runner(TASK, _config())
    result = runner.check("broken", BROKEN)
    assert result.status == "compile_error"
    assert result.latency_ms is None
    assert "SyntaxError" in result.message


def test_accuracy_fail_has_no_latency():
    runner = Runner(TASK, _config())
    result = runner.check("wrong_axis", WRONG_AXIS)
    assert result.status == "accuracy_fail"
    assert result.latency_ms is None
    assert result.max_abs_err is not None


def test_ok_source_returns_latency_and_profile():
    runner = Runner(TASK, _config())
    result = runner.check("stable", STABLE)
    assert result.status == "ok"
    assert result.latency_ms is not None
    assert result.stats is not None
    assert result.stats["samples"] == 1
    profile = runner.profile("stable", STABLE)
    assert isinstance(profile, list)


def test_os_exit_is_caught_as_crash():
    runner = Runner(TASK, _config())
    result = runner.check("exit", EXIT)
    assert result.status == "crash"
    assert result.latency_ms is None
