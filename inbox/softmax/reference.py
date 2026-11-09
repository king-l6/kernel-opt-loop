"""算法同学交给系统的函数。

系统做两件事：把它的输出当成对照，把它的耗时当成基线。
改写必须和这个函数的输出一致，并且更快。
"""

def softmax(x):
    out = torch.empty_like(x)
    rows = x.reshape(-1, x.shape[-1])
    dest = out.reshape(-1, x.shape[-1])
    for i in range(rows.shape[0]):
        row = rows[i]
        shifted = row - row.max()
        exp = torch.exp(shifted)
        dest[i] = exp / exp.sum()
    return out
