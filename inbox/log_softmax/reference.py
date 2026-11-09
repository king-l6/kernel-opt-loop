"""算法同学交给系统的参考实现：逐行、正确、慢。"""

def log_softmax(x):
    out = torch.empty_like(x)
    rows = x.reshape(-1, x.shape[-1])
    dest = out.reshape(-1, x.shape[-1])
    for i in range(rows.shape[0]):
        row = rows[i]
        m = row.max()
        shifted = row - m
        dest[i] = shifted - torch.log(torch.exp(shifted).sum())
    return out
