"""离线演示用的候选源码目录。

每一份都定义与任务同名的入口函数。`ScriptedProposer` 按顺序提交，
用来在没有 LLM key 时把整条闭环跑通；它不是“Agent 会思考”的证据，
真正的生成器是 proposers.LLMProposer。
"""

from __future__ import annotations


def _item(name, title, source, hypothesis, expected_gain=1.0, requires=None):
    return {
        "name": name,
        "title": title,
        "source": source,
        "hypothesis": hypothesis,
        "expected_gain": expected_gain,
        "requires": requires,
    }


SOFTMAX = [
    _item(
        "broken_syntax",
        "括号没写完",
        r'''
def softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m
    return e / e.sum(dim=-1, keepdim=True)
''',
        "先让代码过语法门；这一版故意留一个未闭合的括号。",
    ),
    _item(
        "wrong_axis",
        "轴用错了",
        r'''
def softmax(x):
    m = x.max(dim=0, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=0, keepdim=True)
''',
        "改成张量写法，但把 reduction 维写成了 dim=0。",
    ),
    _item(
        "unstable",
        "没减最大值",
        r'''
def softmax(x):
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)
''',
        "轴修对了，但没有减最大值，大数输入会溢出。",
    ),
    _item(
        "stable_vec",
        "改成向量化",
        r'''
def softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)
''',
        "减最大值保证数值稳定，同时用一次整块张量运算替掉逐行循环。",
        5.0,
    ),
    _item(
        "redundant",
        "多抄了一遍",
        r'''
def softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    y = e / e.sum(dim=-1, keepdim=True)
    acc = torch.empty_like(y)
    rows = y.reshape(-1, y.shape[-1])
    dest = acc.reshape(-1, y.shape[-1])
    for i in range(rows.shape[0]):
        dest[i] = rows[i]
    return acc
''',
        "想验证“正确但更慢”的版本不会被替换。",
        0.8,
    ),
    _item(
        "triton_softmax",
        "改成 Triton",
        r'''
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(out_ptr, in_ptr, stride_m, stride_n, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride_m + cols * stride_n, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * stride_m + cols * stride_n, num / den, mask=mask)

def softmax(x):
    n_cols = x.shape[-1]
    block = triton.next_power_of_2(n_cols)
    view = x.reshape(-1, n_cols)
    out = torch.empty_like(view)
    _softmax_kernel[(view.shape[0],)](
        out, view, view.stride(0), view.stride(1), n_cols, block
    )
    return out.reshape(x.shape)
''',
        "把整行 softmax 收进一个 Triton kernel，减少中间张量和调度开销。",
        8.0,
        requires="cuda+triton",
    ),
]


LOG_SOFTMAX = [
    _item(
        "broken_syntax",
        "括号没写完",
        r'''
def log_softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    shifted = x - m
    return shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True)
''',
        "先过语法门。",
    ),
    _item(
        "wrong_axis",
        "轴用错了",
        r'''
def log_softmax(x):
    m = x.max(dim=0, keepdim=True).values
    shifted = x - m
    return shifted - torch.log(torch.exp(shifted).sum(dim=0, keepdim=True))
''',
        "改成张量写法，但 reduction 维写错。",
    ),
    _item(
        "unstable",
        "没减最大值",
        r'''
def log_softmax(x):
    return x - torch.log(torch.exp(x).sum(dim=-1, keepdim=True))
''',
        "省略最大值平移，大数输入会溢出。",
    ),
    _item(
        "stable_vec",
        "改成向量化",
        r'''
def log_softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    shifted = x - m
    return shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
''',
        "减最大值后用整块张量运算替掉逐行循环。",
        5.0,
    ),
    _item(
        "redundant",
        "多抄了一遍",
        r'''
def log_softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    shifted = x - m
    y = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    acc = torch.empty_like(y)
    rows = y.reshape(-1, y.shape[-1])
    dest = acc.reshape(-1, y.shape[-1])
    for i in range(rows.shape[0]):
        dest[i] = rows[i]
    return acc
''',
        "验证更慢的正确版本不会被采纳。",
        0.8,
    ),
]


LAYERNORM = [
    _item(
        "broken_syntax",
        "括号没写完",
        r'''
def layernorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5
''',
        "先过语法门。",
    ),
    _item(
        "wrong_axis",
        "轴用错了",
        r'''
def layernorm(x):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5)
''',
        "用了 dim=0，输出形状和归一化对象都错了。",
    ),
    _item(
        "unstable",
        "丢了 eps",
        r'''
def layernorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var)
''',
        "归一化分母没有 eps，全零输入会除零。",
    ),
    _item(
        "stable_vec",
        "改成向量化",
        r'''
def layernorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5)
''',
        "一次算完 mean/var，eps 保证零输入稳定。",
        5.0,
    ),
    _item(
        "redundant",
        "多抄了一遍",
        r'''
def layernorm(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(var + 1e-5)
    acc = torch.empty_like(y)
    rows = y.reshape(-1, y.shape[-1])
    dest = acc.reshape(-1, y.shape[-1])
    for i in range(rows.shape[0]):
        dest[i] = rows[i]
    return acc
''',
        "正确但多了一次 Python 复制，应该被拒绝。",
        0.8,
    ),
]


RMSNORM = [
    _item(
        "broken_syntax",
        "括号没写完",
        r'''
def rmsnorm(x):
    denom = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    return x / denom
''',
        "先过语法门。",
    ),
    _item(
        "wrong_axis",
        "轴用错了",
        r'''
def rmsnorm(x):
    denom = torch.sqrt(x.pow(2).mean(dim=0, keepdim=True) + 1e-6)
    return x / denom
''',
        "按列求 RMS，归一化维度错了。",
    ),
    _item(
        "unstable",
        "均值当成了均方",
        r'''
def rmsnorm(x):
    return x / torch.sqrt(x.mean(dim=-1, keepdim=True))
''',
        "把均方根写成了均值，零输入会除零。",
    ),
    _item(
        "stable_vec",
        "改成向量化",
        r'''
def rmsnorm(x):
    denom = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return x / denom
''',
        "一次算完均方和，eps 保证零输入稳定。",
        5.0,
    ),
    _item(
        "redundant",
        "多抄了一遍",
        r'''
def rmsnorm(x):
    denom = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    y = x / denom
    acc = torch.empty_like(y)
    rows = y.reshape(-1, y.shape[-1])
    dest = acc.reshape(-1, y.shape[-1])
    for i in range(rows.shape[0]):
        dest[i] = rows[i]
    return acc
''',
        "正确但多了一次 Python 复制，应该被拒绝。",
        0.8,
    ),
]


CATALOGS = {
    "softmax": SOFTMAX,
    "log_softmax": LOG_SOFTMAX,
    "layernorm": LAYERNORM,
    "rmsnorm": RMSNORM,
}
