"""算法同学交给系统的参考实现：逐行、正确、慢。"""

def layernorm(x):
    out = torch.empty_like(x)
    rows = x.reshape(-1, x.shape[-1])
    dest = out.reshape(-1, x.shape[-1])
    for i in range(rows.shape[0]):
        row = rows[i]
        mean = row.mean()
        var = ((row - mean) ** 2).mean()
        dest[i] = (row - mean) / torch.sqrt(var + 1e-5)
    return out
