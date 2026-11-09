"""算法同学交给系统的参考实现：逐行、正确、慢。"""

def rmsnorm(x):
    out = torch.empty_like(x)
    rows = x.reshape(-1, x.shape[-1])
    dest = out.reshape(-1, x.shape[-1])
    for i in range(rows.shape[0]):
        row = rows[i]
        dest[i] = row / torch.sqrt((row ** 2).mean() + 1e-6)
    return out
