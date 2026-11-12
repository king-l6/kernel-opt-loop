# kernel-opt-loop

一个能在本机跑完的**算子优化 Agent 实验台**。

Agent 每轮只拿到一份结构化反馈：上一轮的门禁结果、人类可读的一句话、当前最优源码与 profile、已尝试过的候选摘要。它交回下一版源码；源码先过静态策略，再进子进程沙箱，按「编译 → 精度 → 测速」三道门判定。只有**正确且比当前最好快 1.1×** 的版本才会替换最优；低于目标、预算用完、连续多轮没提升就停。每一轮的源码、门禁结果、反馈和 profile 都写进轨迹，可以重放核对。

这台机器没有加速卡，默认在 CPU 上测；设备、线程数、超时、预算都在任务里声明。CUDA/Triton 的候选目录已经写好，换到有卡的机器上会自动进入候选。

---

## 对应岗位在做什么

| JD 里的能力 | 这个项目里的位置 |
|---|---|
| 任务分析、代码生成、编译执行、正确性、性能、迭代 | `scheduler.py` + `worker.py` 的完整闭环 |
| 代码沙箱、失败恢复、轨迹追踪、可复现实验 | `sandbox.py`、`trace.py`、`loop.replay()` |
| Profiler、GPU/NPU、算子生成 | `worker._profile`、`catalogs.triton_softmax`、`--device` |
| 任务规划、上下文管理、反馈利用、预算 | `proposers.py`、`feedback.py`、`Config` |
| 可验证任务、评测集、优化经验 | `inbox/*`、`suite.py`、`report.py`、`metrics.py` |

它对应 Kernaut V1 的“单算子无人值守闭环”骨架；V2 的整网联合优化不在这里，但 `DeviceAdapter`/任务格式留了扩展位。

---

## 架构

```
inbox/<task>/
├── task.json          # 入口、输入形状、精度用例、目标、预算
└── reference.py       # 算法同学给的参考实现（对照 + 基线）
        │
        ▼
   intake.py           # 只解析，不执行 reference.py
        │
        ▼
 scheduler.py ──────── 门禁顺序、采纳线、patience、查重、预算、停止
        │  ▲
        │  └────────── feedback.py   结构化反馈 + 一句话
        ▼
 proposers.py ──────── ScriptedProposer（离线） / LLMProposer（OpenAI 兼容 / Anthropic）
        │ source
        ▼
 policy.py ─────────── 允许的 import、禁止的文件/网络/进程调用、入口函数
        │
        ▼
 sandbox.py ────────── 子进程 + 独立进程组 + 超时 SIGKILL + rlimit + 临时 HOME
        │
        ▼
 worker.py ─────────── compile → accuracy（多组边界输入）→ benchmark（p50/p90/CV）→ profile
        │
        ▼
 traces/*.json ─────── replay / report / serve
```

关键原则：**生成器无权改评测逻辑**。它只能返回源码；门禁、采纳线和停止条件都是普通程序。

---

## 快速开始

需要 Python 3.12（系统默认的 3.14 目前没有合适的 PyTorch 轮子）。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. 离线跑通完整闭环（不需要 LLM key）
python run.py inbox/softmax --proposer scripted

# 2. 跑四个任务并汇总
python suite.py inbox --proposer scripted
python report.py traces/suite

# 3. 重放一条轨迹，核对正确性判定
python run.py --replay traces/latest.json

# 4. 本地控制台（可发起真实运行、实时看逐轮门禁、随时停止）
python serve.py
# 浏览器打开 http://127.0.0.1:8765/web/
```

### 接真实 LLM

不依赖任何 SDK，直接用 `urllib` 调两种协议：

```bash
# OpenAI 兼容（DeepSeek / Moonshot / 本地 vLLM 都行）
export KOPT_LLM_PROVIDER=openai
export KOPT_LLM_BASE_URL=https://api.deepseek.com/v1
export KOPT_LLM_API_KEY=sk-...
export KOPT_LLM_MODEL=deepseek-chat

# 或者 Anthropic
export KOPT_LLM_PROVIDER=anthropic
export KOPT_LLM_API_KEY=sk-ant-...
export KOPT_LLM_MODEL=claude-sonnet-5-5

python run.py inbox/softmax --proposer llm --best-of 3
```

`auto`（默认）会在检测到 key 时走 LLM，否则退回离线目录：

```bash
python run.py inbox/softmax                       # auto
python run.py inbox/softmax --proposer scripted   # 强制离线
```

`--best-of 3` 会并行生成 3 个候选，串行评测，避免并行测速互相污染。

### 网页控制台

`python serve.py` 只绑 `127.0.0.1`，页面可以：

- 选任务、生成器、best-of、预算，发起一次真实运行；
- 每轮结束时通过 `--progress` 的 `PROGRESS` 行实时更新轮次、门禁状态和耗时；
- 运行中随时停止（`SIGTERM` 整个进程组，3 秒后 `SIGKILL`），已停止的运行会标记为 `cancelled`；
- 结束后自动加载轨迹，看每轮步骤、profile、候选源码和与 `reference.py` 的 diff；
- 「最近的运行」保留本次服务启动后的运行记录（包括失败和已停止的），历史轨迹按任务取最新一条汇总。

对应的 HTTP 接口：

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/api/tasks` | 任务列表与 LLM 可用性 |
| `GET` | `/api/traces` | 轨迹列表 |
| `GET` | `/api/loop/run?path=…&diff=1` | 读一条轨迹，附带 diff |
| `GET` | `/api/report` | 每个任务最新轨迹的汇总 |
| `POST` | `/api/run` | 启动运行 `{task, proposer, best_of, budget}` |
| `GET` | `/api/run?id=…` | 轮询运行状态、日志、逐轮进度 |
| `POST` | `/api/run/stop` | 停止运行 `{id}` |
| `GET` | `/api/runs` | 本次服务启动后的运行记录 |

---

## 任务格式

```json
{
  "name": "softmax",
  "entry": "softmax",
  "card": "本机 CPU",
  "dtype": "float32",
  "target_ms": 1.0,
  "budget": 8,
  "patience": 4,
  "repeats": 9,
  "warmup": 3,
  "speedup_required": 1.1,
  "timeout_s": 30,
  "threads": 4,
  "bench": {
    "args": [{"shape": [512, 1024], "dist": "randn", "seed": 1}]
  },
  "accuracy": [
    {"args": [{"shape": [8, 32], "dist": "randn", "seed": 0}]},
    {"args": [{"shape": [4, 16, 32], "dist": "randn", "seed": 0}]},
    {"args": [{"shape": [1, 3], "dist": "constant", "value": 1000.0}]},
    {"args": [{"shape": [2, 4], "dist": "zeros"}]},
    {"args": [{"shape": [4, 8], "dist": "randn", "seed": 2, "transpose": true}]}
  ]
}
```

- `dist`：`randn` / `uniform` / `zeros` / `ones` / `full` / `constant` / `randint`。
- `transpose: true` 生成非连续输入，专门抓 `reshape`/stride 类错误。
- 旧的 `{"shape": [...]}` 格式仍然兼容，会自动补一组边界用例。
- `reference.py` 里不需要 `import torch`，worker 会提供；写了也可以。

---

## Agent 每轮看到什么

```json
{
  "round": 3,
  "status": "accuracy_fail",
  "error_kind": "non_finite",
  "message": "case 2: non-finite output",
  "latency_ms": null,
  "best_ms": 3.125,
  "baseline_ms": 3.125,
  "target_ms": 1.0,
  "speedup_vs_best": null,
  "tried": ["broken_syntax", "wrong_axis", "unstable"],
  "profile": [{"name": "aten::exp", "us": 812.0, "pct": 62.0}],
  "hints": ["先做数值稳定化：softmax/log_softmax 减最大值，norm 加 eps。"],
  "text": "结果不一致：case 2: non-finite output。先修正确性，暂不测速。"
}
```

LLM 拿到的还包括任务规格、`reference_source`、当前最优源码和最近 6 轮摘要；**不塞全量历史**，避免上下文膨胀。

---

## 门禁与停止

1. **策略**：只允许 `torch` / `triton` / `math` / `typing` / `numpy`；禁止 `open`、`eval`、`exec`、`import os/subprocess/socket/...`；必须定义入口函数。
2. **编译**：沙箱里 `compile` + `exec`，语法/加载错误直接反馈，不测速。
3. **精度**：多组输入（随机、大数、全零、非连续）逐组对照；形状、dtype、有限性、`allclose` 全过才继续。
4. **测速**：warmup + 9 次重复，取 p50，同时记录 p90、min、max、CV；CUDA 上用 `torch.cuda.Event`。
5. **采纳**：`当前最好 / 本次` ≥ `speedup_required`（默认 1.1）才替换；被采纳后再采一次 profile。
6. **停止**：`target` / `budget` / `patience` / `no_proposal` / `token_budget` / `proposer_error`。

查重按 AST 归一化后的 SHA-256 判定：改注释、改空白不算新候选，不会重复消耗沙箱。

---

## 沙箱边界（诚实版）

`worker.py` 是独立子进程，`sandbox.py` 负责：

- 独立进程组，超时 `SIGKILL` 整组；
- 一次性临时目录作为 cwd / HOME / TMPDIR；
- 环境变量白名单重建，不继承 API key、代理、项目路径；
- `RLIMIT_CPU`、`RLIMIT_FSIZE`，worker 内再尝试 `RLIMIT_AS`；
- 结果走独立 JSON 文件，不解析 stdout（候选 `print` 不会污染协议）。

**它不是容器级隔离**：候选如果主动 `fork` 到新会话、或通过网络外连，进程级限制拦不住。要跑完全不可信的模型生成代码，把 `sandbox.run()` 换成 Docker/nsjail 实现，接口不变。

---

## 指标与可复现

```bash
python report.py traces/suite
```

输出每个任务的 `best / speedup / target / experiments / tokens`，以及：

- `success@target`：达标任务占比；
- `geomean_speedup`：跨任务几何平均加速；
- `experiments_per_success`、`tokens_per_success`：单位成功成本；
- `accept_rate`：候选被采纳比例。

`run.py --replay` 不追求延迟小数点一致，而是核对**每轮的正确性判定**是否仍成立，并对超过 3× 的延迟漂移给警告。轨迹里带环境指纹：Python/torch/numpy 版本、平台、CPU 数、torch 线程数、git sha。

---

## 四个内置任务

| 任务 | 基线慢在哪 | 典型候选路径 |
|---|---|---|
| `softmax` | 逐行 `exp/sum` | 语法 → 轴错 → 溢出 → 向量化 |
| `log_softmax` | 逐行 `log(exp)` | 同上 |
| `layernorm` | 逐行 mean/var | 语法 → 轴错 → 丢 eps → 向量化 |
| `rmsnorm` | 逐行均方 | 语法 → 轴错 → 均值当均方 → 向量化 |

离线目录在 `catalogs.py`，只用于没有 key 时跑通闭环；它不是“Agent 会思考”的证据。真正的生成器是 `proposers.LLMProposer`。

---

## 目录

```
intake.py       任务读取与旧格式兼容
policy.py       静态策略
sandbox.py      父进程侧沙箱
worker.py       子进程侧编译/精度/测速/profile
gates.py        Runner 与 Eval
feedback.py     结构化反馈 + 一句话
proposers.py    ScriptedProposer / LLMProposer / LLMClient
catalogs.py     离线候选目录
scheduler.py    门禁顺序、预算、采纳、停止、轨迹记录
trace.py        环境指纹与落盘
loop.py         Config / run / replay
metrics.py      指标
report.py       轨迹汇总 CLI
suite.py        任务集 CLI
run.py          单任务 CLI
serve.py        本地控制台 + 只读 API
web/index.html  控制台页面
inbox/*         任务
traces/*        轨迹
```

---

## 已知限制与下一步

- **更强隔离**：Docker/nsjail 适配器；网络默认拒绝。
- **真 GPU**：`--device cuda` 已支持 CUDA events 计时和 Triton 候选；需要一台有卡机器做一次真实验收。
- **国产芯片**：把 `device` 与 worker 的输入/测速分支抽象成 `DeviceAdapter`，接 CANN/Ascend C；当前只有接口位。
- **整网与推理配置**：下一步是算子 + 显存 + 执行图 + vLLM/SGLang 配置的联合搜索，而不是只改单个 kernel。
- **评测集**：扩充到 KernelBench/TritonBench 子集，按算子类别统计成功率。
- **经验回流**：把成功轨迹提炼成技巧库，下一轮作为提示注入；当前只有历史摘要。

---

## 一句话讲这个项目

> 我把算子优化做成了一个可复现的实验系统：候选代码进沙箱，编译/精度/性能三道门，只有更快才替换最优，轨迹能重放，效果用 success@target 和几何平均加速衡量。
