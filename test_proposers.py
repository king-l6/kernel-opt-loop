import json

from proposers import (
    Completion,
    LLMProposer,
    ScriptedProposer,
    _extract_json,
    source_signature,
)


class FakeClient:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, messages, max_tokens=4096, temperature=0.7):
        self.calls.append(messages)
        return Completion(self.texts.pop(0), {"total_tokens": 12})


def _ctx():
    return {
        "task_spec": {"name": "softmax", "entry": "softmax"},
        "reference_source": "def softmax(x):\n    return x\n",
        "current_best": {"name": "baseline", "source": "def softmax(x):\n    return x\n", "latency_ms": 10.0},
        "baseline_ms": 10.0,
        "target_ms": 1.0,
        "feedback": {"status": "baseline", "text": "基线耗时 10.000 ms。", "profile": []},
        "history": [],
        "round": 1,
        "budget_left": 8,
        "best_of": 1,
        "device": "cpu",
        "triton_ready": False,
    }


def test_extract_json_handles_markdown_fences():
    data = _extract_json('```json\n{"source": "x", "hypothesis": "y"}\n```')
    assert data == {"source": "x", "hypothesis": "y"}


def test_llm_proposer_returns_patch_and_usage():
    body = json.dumps(
        {
            "hypothesis": "向量化",
            "source": "def softmax(x):\n    return torch.softmax(x, dim=-1)\n",
            "expected_gain": 3.0,
        }
    )
    client = FakeClient([body])
    patches = LLMProposer(client, "softmax").propose(_ctx(), n=1)
    assert len(patches) == 1
    assert "torch.softmax" in patches[0].source
    assert patches[0].usage["total_tokens"] == 12
    assert client.calls[0][0]["role"] == "system"


def test_llm_proposer_repairs_bad_json_once():
    good = json.dumps({"source": "def softmax(x):\n    return x\n", "hypothesis": "ok"})
    client = FakeClient(["这不是 JSON", good])
    patches = LLMProposer(client, "softmax", repair_retries=1).propose(_ctx(), n=1)
    assert len(patches) == 1
    assert len(client.calls) == 2


def test_llm_proposer_accepts_give_up():
    client = FakeClient([json.dumps({"source": None, "hypothesis": "没有空间"})])
    proposer = LLMProposer(client, "softmax")
    assert proposer.propose(_ctx(), n=1) == []
    assert "没有空间" in proposer.last_error


def test_scripted_proposer_skips_cuda_only_candidates_on_cpu():
    catalog = [
        {"name": "cuda", "title": "cuda", "source": "def softmax(x):\n    return x\n", "requires": "cuda+triton"},
        {"name": "cpu", "title": "cpu", "source": "def softmax(x):\n    return x + 1\n"},
    ]
    proposer = ScriptedProposer(catalog, "softmax", device="cpu", triton_ready=False)
    patches = proposer.propose(_ctx())
    assert patches and patches[0].name == "cpu"
    assert proposer.propose(_ctx()) == []


def test_source_signature_ignores_comments_and_whitespace():
    left = "def f(x):\n    return x\n"
    right = "# 注释\ndef f( x ):\n\n    return x  # 尾注释\n"
    assert source_signature(left) == source_signature(right)
