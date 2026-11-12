import ast

import pytest

from catalogs import CATALOGS
from policy import check_source


def test_valid_source_passes():
    kind, message = check_source("def softmax(x):\n    return x\n", "softmax")
    assert (kind, message) == ("", "")


def test_syntax_error_is_reported_as_syntax():
    kind, message = check_source("def softmax(x:\n    return x\n", "softmax")
    assert kind == "syntax"
    assert "SyntaxError" in message


def test_forbidden_import_is_rejected():
    kind, message = check_source("import os\ndef softmax(x):\n    return os.getcwd()\n", "softmax")
    assert kind == "policy"
    assert "os" in message


def test_forbidden_call_is_rejected():
    kind, message = check_source("def softmax(x):\n    return eval('x')\n", "softmax")
    assert kind == "policy"
    assert "eval" in message


def test_torch_compile_is_allowed():
    source = "def softmax(x):\n    return torch.compile(lambda y: torch.softmax(y, dim=-1))(x)\n"
    assert check_source(source, "softmax") == ("", "")


def test_future_import_is_allowed():
    source = "from __future__ import annotations\n\ndef softmax(x):\n    return x\n"
    assert check_source(source, "softmax") == ("", "")


def test_missing_entry_is_rejected():
    kind, message = check_source("def other(x):\n    return x\n", "softmax")
    assert kind == "policy"
    assert "softmax" in message


def test_empty_source_is_rejected():
    kind, message = check_source("", "softmax")
    assert kind == "policy"


def test_module_attribute_assignment_is_rejected():
    source = "def softmax(x):\n    torch.allclose = lambda *a, **k: True\n    return x\n"
    kind, message = check_source(source, "softmax")
    assert kind == "policy"
    assert "属性" in message


def test_tensor_class_patching_is_rejected():
    source = "def softmax(x):\n    torch.Tensor.abs = lambda self: None\n    return x\n"
    assert check_source(source, "softmax")[0] == "policy"


def test_setattr_and_vars_are_rejected():
    for body in [
        "    setattr(torch, 'allclose', lambda *a, **k: True)\n    return x\n",
        "    vars(torch)['allclose'] = lambda *a, **k: True\n    return x\n",
    ]:
        kind, _ = check_source("def softmax(x):\n" + body, "softmax")
        assert kind == "policy"


def test_ordinary_tensor_assignments_still_pass():
    source = (
        "def softmax(x):\n"
        "    y = x - x.max(dim=-1, keepdim=True).values\n"
        "    y = torch.exp(y)\n"
        "    return y / y.sum(dim=-1, keepdim=True)\n"
    )
    assert check_source(source, "softmax") == ("", "")


def test_catalog_sources_parse_except_intentional_broken_one():
    for name, items in CATALOGS.items():
        for item in items:
            if item["name"] == "broken_syntax":
                with pytest.raises(SyntaxError):
                    ast.parse(item["source"])
            else:
                ast.parse(item["source"])
