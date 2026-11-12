# kernel-opt-loop

一个能在本机跑完的**算子优化 Agent 实验台**。

Agent 每轮拿到一份结构化反馈：上一轮的门禁结果、人类可读的一句话、当前最优源码与 profile、已尝试过的候选摘要。它可以在受限预算内先**试跑**候选、读完整错误和 profile，再交回下一版源码；源码先过静态策略，再进子进程沙箱，按「编译 → 精度 → 测速」三道门判定。只有**正确且比当前最好快 1.1×** 的版本才会替换最优；低于目标、预算用完、连续多轮没提升就停。每一轮的源码、门禁结果、反馈、profile 和设备带宽都写进轨迹，可以重放核对。

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
fleet.py ──────────── 任务 × 设备矩阵（同卡串行、跨卡并行，不可用设备跳过）
        │
        ▼
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
        │  ├────────── memory.py     跨运行经验：warm start、lessons、failures
        │  └────────── feedback.py   结构化反馈 + 一句话
        ▼
 proposers.py ──────── ScriptedProposer / LLMProposer / AgentProposer（带工具循环）
        │ source                 ▲
        │                        │ probe（试跑、profile）
        │                  probes.py ── 独立预算 + AST 缓存，不更新最优
        ▼
 policy.py ─────────── 允许的 import、禁止的文件/网络/进程调用、入口函数
        │
        ▼
 sandbox.py ────────── 子进程 + 独立进程组 + 超时 SIGKILL + rlimit + 临时 HOME
        │
        ▼
 worker.py ─────────── compile → accuracy（多组边界输入）→ benchmark（p50/p90/CV）→ profile
        │              （同时回报最小 I/O 字节数、CUDA 时钟）
        ▼
 devices.py ────────── 认出这块卡、纸面峰值、有效带宽、单次 I/O 下界
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

# 5. 把任务集铺到多块设备上（同卡串行、跨卡并行）
python fleet.py inbox --devices cpu,cuda:0 --budget 2

# 6. 跨运行记忆（默认开启，存在 ./memory；下一次从上次最优继续）
python run.py inbox/softmax --memory-dir memory
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

`auto`（默认）会在检测到 key 时走 Agent（可试跑），否则退回离线目录：

```bash
python run.py inbox/softmax                       # auto -> agent
python run.py inbox/softmax --proposer agent --probe-budget 3 --agent-steps 6
python run.py inbox/softmax --proposer llm         # 单次提交，作为对照组
python run.py inbox/softmax --proposer scripted    # 强制离线
```

`--best-of 3` 会并行生成 3 个候选，串行评测，避免并行测速互相污染。

### Agent 工具循环

`AgentProposer` 每轮最多走 `agent_steps` 步，每一步只输出一个 JSON 动作：

```json
{"action": "probe", "source": "...", "profile": true}
{"action": "submit", "source": "...", "hypothesis": "...", "expected_gain": 1.2}
{"action": "give_up", "hypothesis": "原因"}
```

- `probe` 走和正式门禁完全相同的编译、精度、测速，返回 `status / message / stderr / latency / profile`；
- 试跑结果按 AST 缓存，同一份源码不重复消耗真机；语法/策略错误不消耗试跑预算；
- `probe_budget` 是整条运行的试跑上限，和 `budget`（提交实验数）分开计费；
- 试跑不更新最优、不写轨迹；是否采纳仍由 scheduler 判定；步数用完时返回试跑过的最好一版。

真实模型的最小验证（本机 CPU，1 次提交）：

```bash
python run.py inbox/softmax --proposer agent --budget 1 --probe-budget 1 --agent-steps 4
# best: torch.softmax(x, dim=-1)  0.214 ms  speedup=13.76x  tokens≈4.2k
```

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
  "agent_steps": 8,
  "probe_budget": 6,
  "device_spec": {
    "name": "Apple M1",
    "peak_bw_gbps": 68.25
  },
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
  "hardware": {
    "device": "Apple M1",
    "gbps": 16.8,
    "bw_pct": 0.25,
    "floor_ms": 0.061,
    "advice": ["按最小 I/O 用到 25% 纸面带宽（下界 0.061 ms…）；先确认是单遍融合…"]
  },
  "hints": ["先做数值稳定化：softmax/log_softmax 减最大值，norm 加 eps。"],
  "text": "结果不一致：case 2: non-finite output。先修正确性，暂不测速。"
}
```

LLM 拿到的还包括任务规格、`device_info`（实际设备与峰值）、`reference_source`、当前最优源码和最近 6 轮摘要；**不塞全量历史**，避免上下文膨胀。

---

## 门禁与停止

1. **策略**：只允许 `torch` / `triton` / `math` / `typing` / `numpy`；禁止 `open`、`eval`、`exec`、`import os/subprocess/socket/...`；必须定义入口函数。
2. **编译**：沙箱里 `compile` + `exec`，语法/加载错误直接反馈，不测速。
3. **精度**：多组输入（随机、大数、全零、非连续）逐组对照；形状、dtype、有限性、`allclose` 全过才继续。
4. **测速**：warmup + 9 次重复，取 p50，同时记录 p90、min、max、CV；CUDA 上用 `torch.cuda.Event`。CV > 0.15 视为测量被瞬时干扰，自动重测一次，避免把噪声当成代码变慢。
5. **采纳**：`当前最好 / 本次` ≥ `speedup_required`（默认 1.1）才替换；被采纳后再采一次 profile。
6. **停止**：`target` / `budget` / `patience` / `no_proposal` / `token_budget` / `proposer_error`。

查重按 AST 归一化后的 SHA-256 判定：改注释、改空白不算新候选，不会重复消耗沙箱。

门禁自己用的原语（`allclose`、`isfinite`、Tensor 的 `shape/dtype`、计时函数）会在候选执行前抓取原始引用；候选改写 `torch.allclose = ...` 之类只影响它自己，不会影响判定。`policy.py` 也会直接拒绝改写模块/类属性的源码。

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

## 设备反馈：不只是「快了几毫秒」

同一份算子在不同卡上的瓶颈不同，所以每轮反馈里都带一份设备视图：

```json
{
  "device": "Apple M1",
  "backend": "cpu",
  "peak_bw_gbps": 68.25,
  "peak_source": "spec",
  "io_bytes": 4194304,
  "gbps": 1.447,
  "bw_pct": 0.021,
  "floor_ms": 0.061,
  "io_headroom": 47.2,
  "cache_resident": false,
  "advice": ["按最小 I/O 只有 2% 纸面带宽（下界 0.061 ms，乐观上限 47.2×，未计中间张量）。如果还有逐行循环或分离的 exp/sum/div，先融合…"]
}
```

- **设备是谁**：CUDA 从 `torch.cuda.get_device_properties` 取名称、SM 数、算力、显存；CPU/MPS 取芯片型号。纸面峰值来自 `devices.py` 的规格表，命中不了就返回 `null`，不猜。
- **有效带宽**：子进程在候选执行前，用参考实现在基准输入上算一遍输出，得到「输入 + 输出」的最小 I/O 字节数；`io_bytes / 延迟` 就是有效带宽。这个数由父进程决定，候选改写 torch 也改不到。
- **单次 I/O 下界**：`io_bytes / 纸面峰值`。这是**乐观下界**——多 pass 的中间张量没算进去，所以它是「理论上最快能到多少」，不是「利用率」。
- **缓存识别**：小 shape 下算出的带宽可能超过 DRAM 峰值（数据落在缓存里），这时标记 `cache_resident`，并提示换更大的 shape 或看 profiler 的实际访存计数。
- **CUDA 时钟**：测速前后各采一次 SM 时钟，降频超过 10% 会体现在 `clocks.drop_pct` 里。

未登记的卡可以在 `task.json` 写 `device_spec`，或运行时加 `--peak-bw`：

```bash
python run.py inbox/softmax --device cuda:0 --peak-bw 1555
```

显式指定的设备不可用时会直接报错，不会悄悄退回 CPU——否则「在 A100 上跑过」的结论会被 CPU 数字污染。

反馈里的 `advice` 会直接进下一轮 prompt，Agent 因此能区分「还没写向量化」和「已经贴着带宽下界、只能减少字节数」这两种情况。

多设备用 `fleet.py`：同一块卡上的任务串行（并发会互相抢带宽），不同卡之间并行；显式指定的设备不可用时跳过并记录，不会悄悄退回 CPU。

```bash
python fleet.py inbox --devices cpu,cuda:0 --budget 2
# task            cpu:Apple M1 68GB/s      cuda:0:不可用
# softmax                   7.47x ✓                  跳过
```

---

## 跨运行经验：同一块卡上不从零开始

单轮内循环只看得到最近几轮，换一次运行就全忘了。`memory.py` 按 `(算子, 设备, 基准形状)` 存一个 JSON（`memory/softmax@apple-m1@512x1024-f32-<hash>.json`）：

- `best`：上次最好的源码、耗时、相对基线加速比；
- `lessons`：被采纳过的改法；
- `failures`：反复出现的错误类型（语法、精度、超时…）和次数；
- `dead_ends`：试过但没变快的方向。

下一次运行开始，scheduler 先拿 `best.source` 直接上机（warm start，算 1 次实验）；过了门禁就把它当当前最优。没超过它也不会把记录删掉——只有真正崩了/精度不过才丢弃，避免一次噪声测量把积累的经验清空。

`lessons / failures / dead_ends` 的压缩视图会作为 `memory` 字段进 prompt，Agent 不用重复踩同样的坑。

```json
{
  "task": "softmax",
  "device": "Apple M1",
  "runs": 2,
  "best": {
    "latency_ms": 0.388,
    "baseline_ms": 3.027,
    "speedup": 7.80,
    "source": "def softmax(x): ..."
  },
  "lessons": [
    {"hypothesis": "减最大值保证数值稳定，同时用一次整块张量运算替掉逐行循环。", "speedup": 7.80}
  ],
  "failures": [
    {"status": "accuracy_fail", "error_kind": "non_finite", "count": 1,
     "message": "case 2: non-finite output"}
  ],
  "dead_ends": []
}
```

实测同一台 M1 连跑两次：第一次 3.03 → 0.388 ms（7.80×）；第二次 warm start 直接 0.390 ms（7.41×，CV 0.024），不用重新摸一遍。

- `--memory-dir` 指定目录；`--no-memory` 关闭（对照实验用）。
- `fleet.py` 把记忆放在输出目录的 `_memory/` 下，按设备分开。
- 换 shape 就是另一份记忆：`512×1024` 的最优实现不保证对 `8×32` 最优。
- 跨卡不迁移：A100 的经验不会搬到 M1，不同卡的瓶颈不同，按 `(算子, 设备)` 隔离是刻意的。

---

## 指标与可复现

```bash
python report.py traces/suite
```

输出每个任务的 `best / speedup / target / experiments / probes / tokens`，以及：

- `success@target`：达标任务占比；
- `geomean_speedup`：跨任务几何平均加速；
- `experiments_per_success`、`tokens_per_success`：单位成功成本；
- `accept_rate`：候选被采纳比例。

`run.py --replay` 不追求延迟小数点一致，而是核对**每轮的正确性判定**是否仍成立，并对超过 3× 的延迟漂移给警告。轨迹里带环境指纹：Python/torch/numpy 版本、平台、CPU 数、torch 线程数、git sha；`device_info` 记录实际设备名和纸面峰值，避免把不同卡的加速比混在一起比较。

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
proposers.py    ScriptedProposer / LLMProposer / AgentProposer / LLMClient
probes.py       Agent 的试跑工具（独立预算 + AST 缓存）
devices.py      设备识别、纸面峰值、有效带宽、单次 I/O 下界
memory.py       跨运行经验：按 (算子, 设备) 存 best / lessons / failures
catalogs.py     离线候选目录
scheduler.py    门禁顺序、预算、采纳、停止、轨迹记录
trace.py        环境指纹与落盘
loop.py         Config / run / replay
metrics.py      指标
report.py       轨迹汇总 CLI
suite.py        任务集 CLI
fleet.py        任务 × 设备矩阵 CLI
run.py          单任务 CLI
serve.py        本地控制台 + 只读 API
web/index.html  控制台页面
test_agent.py   Agent 工具循环与试跑预算的测试
test_devices.py 设备识别与带宽下界的测试
test_memory.py  跨运行记忆与 warm start 的测试
test_fleet.py   多设备矩阵的测试
inbox/*         任务
traces/*        轨迹
```

---

## 从单算子到「100 算子 × 多卡」

目标不是单算子演示，而是让每个算子在不同加速卡上都能根据真机反馈持续变快。当前仓库完成的是**单卡内循环**；往外扩需要下面几层，接口已经按这个方向留位：

| 层 | 现在 | 下一步 |
|---|---|---|
| 生成与验证 | Agent 可试跑、看 profile、再提交；门禁不可绕过 | 原生 tool-use / 多文件编辑；编译错误全文回传；失败方向作为下一轮硬约束 |
| 设备反馈 | 设备识别、纸面峰值、最小 I/O 有效带宽、单次 I/O 下界、缓存识别、CUDA 时钟 | 真实 DRAM 计数与 occupancy（`ncu`/CUPTI）、ROCm/Ascend 适配、按卡归一化基线、降频进采纳判定 |
| 任务规模 | 4 个归一化算子，固定 shape | 100 算子的任务生成器：reference、随机/隐藏精度用例、shape/dtype sweep、厂商库基线（cuBLAS/cuDNN） |
| 调度与并发 | `fleet.py`：任务 × 设备矩阵，同卡串行、跨卡并行，不可用设备跳过 | 常驻队列、远程 worker、断点续跑、按卡配额 |
| 经验回流 | 按 `(算子, 设备, 形状)` 存 best / lessons / failures / dead_ends，下次 warm start + 注入 prompt | 跨卡迁移、按相关性检索排序、经验过期淘汰 |
| 成本与统计 | token 预算、单次中位数、1.1× 采纳线、CV>0.15 重测 | prompt caching、置信区间采纳、防 memoization、隐藏用例 |

**关键原则不变**：Agent 负责提出和试错，scheduler 负责预算、门禁和判定。给 Agent 更多工具，不等于让它改评测。

已知限制：设备反馈用的是纸面峰值和「最小 I/O」下界，不是硬件计数器，所以小 shape 会命中缓存、多 pass 会被低估；经验存本地 JSON（单机、不做分布式同步）、噪声重测只能减轻瞬时干扰而不能替代置信区间采纳、进程级沙箱（不是容器）、CUDA/Triton 路径需要在有卡机器上验收、真实模型的动作协议目前是 JSON 而不是原生 tool call。

---

## 一句话讲这个项目

> 我把算子优化做成了一个可复现、可扩展的内循环 Agent：它在真机沙箱里试跑、读 profile、改代码，门禁决定是否采纳，轨迹能重放；目标是把同一套反馈驱动机制铺到 100 种算子和多种加速卡上。
