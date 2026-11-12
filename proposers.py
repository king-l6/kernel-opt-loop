"""候选生成器。离线脚本和真 LLM 走同一个接口。

接口只有一句：
    propose(ctx, n) -> list[Patch]

ctx 是上一轮的完整上下文（结构化反馈、当前最优源码、profile、历史摘要）。
生成器看不到沙箱内部，也拿不到任何写文件的权限。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from catalogs import CATALOGS
from feedback import compact_history


@dataclass
class Patch:
    name: str
    title: str
    source: str | None
    hypothesis: str = ""
    expected_gain: float | None = None
    usage: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class ScriptedProposer:
    """按目录顺序提交候选。只用于离线跑通闭环，不冒充 LLM。"""

    kind = "scripted"

    def __init__(self, catalog, entry: str, device: str = "cpu", triton_ready: bool = False):
        self.catalog = list(catalog)
        self.entry = entry
        self.device = device
        self.triton_ready = triton_ready
        self.tried: set[str] = set()

    def propose(self, ctx: dict, n: int = 1) -> list[Patch]:
        for item in self.catalog:
            name = item["name"]
            if name in self.tried:
                continue
            requires = item.get("requires")
            if requires == "cuda+triton" and not (self.device == "cuda" and self.triton_ready):
                continue
            self.tried.add(name)
            return [
                Patch(
                    name=name,
                    title=item.get("title", name),
                    source=item["source"],
                    hypothesis=item.get("hypothesis", ""),
                    expected_gain=item.get("expected_gain"),
                )
            ]
        return []


# --------------------------------------------------------------------------
# LLM 客户端：OpenAI 兼容 / Anthropic 两种协议，不依赖第三方 SDK
# --------------------------------------------------------------------------


class LLMError(RuntimeError):
    """kind: transient（5xx/429/超时，可重试）/ parse（响应格式）/ fatal（配置或鉴权）"""

    def __init__(self, message: str, kind: str = "fatal"):
        super().__init__(message)
        self.kind = kind


@dataclass
class Completion:
    text: str
    usage: dict = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self.provider = (provider or os.environ.get("KOPT_LLM_PROVIDER") or "").lower()
        if not self.provider:
            self.provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
        self.base_url = (base_url or os.environ.get("KOPT_LLM_BASE_URL") or "").rstrip("/")
        # 兼容 Claude Code 常见的内网网关配置：ANTHROPIC_BASE_URL / OPENAI_BASE_URL
        if not self.base_url and self.provider == "anthropic":
            self.base_url = (os.environ.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
        if not self.base_url and self.provider != "anthropic":
            self.base_url = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            self.base_url = (
                "https://api.anthropic.com"
                if self.provider == "anthropic"
                else "https://api.openai.com/v1"
            )
        self.api_key = (
            api_key
            or os.environ.get("KOPT_LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY")
            or ""
        )
        self.model = model or os.environ.get("KOPT_LLM_MODEL") or (
            "claude-sonnet-5-5" if self.provider == "anthropic" else "gpt-4o-mini"
        )
        self.timeout = timeout
        self.retries = retries

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.7) -> Completion:
        if not self.api_key:
            raise LLMError("没有配置 LLM key：设置 KOPT_LLM_API_KEY，或用 --proposer scripted")
        payload = self._payload(messages, max_tokens, temperature)
        headers = self._headers()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(
                    self._url(),
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                try:
                    return self._parse(body)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise LLMError(f"响应解析失败：{exc}", "parse") from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:800]
                transient = exc.code in {408, 409, 429, 500, 502, 503, 504}
                last_error = LLMError(
                    f"HTTP {exc.code}: {detail}",
                    "transient" if transient else "fatal",
                )
                if not transient:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LLMError(f"连接失败：{exc}", "transient")
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise last_error or LLMError("LLM 调用失败")

    def _url(self) -> str:
        if self.provider == "anthropic":
            return f"{self.base_url}/v1/messages"
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict:
        if self.provider == "anthropic":
            return {
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    def _payload(self, messages, max_tokens, temperature) -> dict:
        if self.provider == "anthropic":
            system = "\n\n".join(item["content"] for item in messages if item["role"] == "system")
            turns = [item for item in messages if item["role"] != "system"]
            return {
                "model": self.model,
                "system": system,
                "messages": turns,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _parse(self, body: str) -> Completion:
        data = json.loads(body)
        if self.provider == "anthropic":
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            usage = data.get("usage") or {}
            usage = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            }
        else:
            choices = data.get("choices") or []
            text = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage") or {}
        return Completion(text=text, usage=usage)


# --------------------------------------------------------------------------
# LLMProposer
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """你是 AI Infra 团队的算子优化 Agent。你的任务是把 reference.py 里的 PyTorch 实现改写成更快的等价实现，并且只能依据真机评测反馈迭代。

硬性约束：
1. 只输出一个 JSON 对象，格式：
   {{"hypothesis": "一句话说明这一版改了什么、为什么更快", "source": "完整 Python 源码", "expected_gain": 1.2}}
2. source 必须定义 {entry}(...)，签名与 reference 一致，返回 Tensor。
3. 只能 import torch、torch.nn.functional、triton、math、typing、numpy；禁止文件、网络、子进程、环境变量、print。
4. 输出形状和 dtype 必须与输入一致。
5. 数值稳定：softmax/log_softmax 先减最大值；norm 必须加 eps。
6. 不要提交与历史相同、或只改注释/变量名的代码。
7. 确实没有改进空间时，输出 {{"source": null, "hypothesis": "原因"}}。

优化顺序：
- 先正确，再性能；
- 去掉逐行 Python 循环，改成整块张量运算；
- 减少中间张量和 copy，融合 exp/sum/div、mean/var；
- CPU 可尝试 torch.compile；CUDA 可尝试 Triton；
- 每次只改一个主要变量，方便归因。

只输出 JSON，不要 markdown 代码围栏，不要额外解释。"""


class LLMProposer:
    kind = "llm"

    def __init__(
        self,
        client: LLMClient,
        entry: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        repair_retries: int = 1,
    ):
        self.client = client
        self.entry = entry
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repair_retries = repair_retries
        self.last_error: str = ""
        self.last_error_kind: str = ""

    def propose(self, ctx: dict, n: int = 1) -> list[Patch]:
        self.last_error = ""
        self.last_error_kind = ""
        messages = build_messages(ctx, self.entry)
        workers = max(1, min(n, 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda i: self._one(messages, i, ctx), range(n)))
        patches: list[Patch] = []
        seen: set[str] = set()
        for patch in results:
            if patch is None or not patch.source:
                continue
            signature = source_signature(patch.source)
            if signature in seen:
                continue
            seen.add(signature)
            patches.append(patch)
        if not patches and any(item is None for item in results) and not self.last_error:
            self.last_error = "模型没有返回可解析的 JSON，或主动放弃"
        if not patches and not self.last_error_kind:
            self.last_error_kind = "parse"
        return patches

    def _one(self, messages: list[dict], index: int, ctx: dict) -> Patch | None:
        attempt_messages = list(messages)
        for attempt in range(self.repair_retries + 1):
            try:
                completion = self.client.complete(
                    attempt_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except LLMError as exc:
                self.last_error = str(exc)
                self.last_error_kind = getattr(exc, "kind", "fatal")
                return None
            try:
                data = _extract_json(completion.text)
            except ValueError as exc:
                self.last_error = f"JSON 解析失败：{exc}"
                self.last_error_kind = "parse"
                if attempt < self.repair_retries:
                    attempt_messages = messages + [
                        {"role": "assistant", "content": completion.text[:2000]},
                        {"role": "user", "content": "上一个输出不是合法 JSON。只输出一个 JSON 对象，不要 markdown。"},
                    ]
                    continue
                return None
            source = data.get("source")
            hypothesis = str(data.get("hypothesis") or "")
            if source is None:
                self.last_error = f"模型放弃：{hypothesis}"
                self.last_error_kind = "declined"
                return None
            if not isinstance(source, str):
                self.last_error = "source 不是字符串"
                self.last_error_kind = "parse"
                return None
            round_no = ctx.get("round", 0)
            title = _short_title(hypothesis) or f"LLM 第 {round_no} 轮"
            return Patch(
                name=f"llm-r{round_no}-{index}",
                title=title,
                source=source,
                hypothesis=hypothesis,
                expected_gain=_optional_float(data.get("expected_gain")),
                usage=completion.usage,
            )
        return None


def build_messages(ctx: dict, entry: str) -> list[dict]:
    feedback = dict(ctx.get("feedback") or {})
    feedback.pop("history", None)
    payload = {
        "task": ctx.get("task_spec"),
        "reference_source": ctx.get("reference_source"),
        "current_best": ctx.get("current_best"),
        "baseline_ms": ctx.get("baseline_ms"),
        "target_ms": ctx.get("target_ms"),
        "device": ctx.get("device"),
        "triton_available": ctx.get("triton_ready"),
        "round": ctx.get("round"),
        "experiments_left": ctx.get("budget_left"),
        "sample_index_hint": ctx.get("best_of", 1),
        "last_feedback": feedback,
        "history": compact_history(ctx.get("history") or [], limit=6),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(entry=entry)},
        {
            "role": "user",
            "content": "以下是本轮上下文。只输出一个 JSON 对象：\n"
            + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def make_proposer(kind: str, task, config):
    device = str(getattr(config, "device", "cpu"))
    if device == "auto":
        from gates import best_device

        device = best_device()
    from gates import triton_ready

    ready = triton_ready(device)
    if kind == "scripted":
        catalog = CATALOGS.get(task.name)
        if not catalog:
            raise RuntimeError(
                f"任务 {task.name} 没有离线候选目录；用 --proposer llm，或在 catalogs.py 里补一份"
            )
        return ScriptedProposer(catalog, task.entry, device, ready), f"scripted:{task.name}"
    if kind == "llm":
        client = LLMClient()
        if not client.available:
            raise RuntimeError(
                "没有 LLM key。设置 KOPT_LLM_API_KEY/KOPT_LLM_BASE_URL/KOPT_LLM_MODEL，"
                "或者用 --proposer scripted"
            )
        label = f"llm:{client.provider}/{client.model}"
        return (
            LLMProposer(client, task.entry, temperature=getattr(config, "temperature", 0.7)),
            label,
        )
    if kind != "auto":
        raise ValueError(f"未知 proposer：{kind}")
    client = LLMClient()
    if client.available:
        label = f"llm:{client.provider}/{client.model}"
        return (
            LLMProposer(client, task.entry, temperature=getattr(config, "temperature", 0.7)),
            label,
        )
    catalog = CATALOGS.get(task.name)
    if not catalog:
        raise RuntimeError(f"没有 LLM key，且任务 {task.name} 没有离线候选目录")
    return ScriptedProposer(catalog, task.entry, device, ready), f"scripted:{task.name}"


def source_signature(source: str) -> str:
    try:
        tree = ast.parse(source)
        payload = ast.dump(tree, include_attributes=False)
    except SyntaxError:
        payload = re.sub(r"\s+", " ", source).strip()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("没有找到 JSON 对象")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("顶层不是对象")
    return data


def _short_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:24]


def _optional_float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
