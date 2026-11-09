"""兼容旧入口：候选源码已迁到 catalogs.py。

新代码请直接 `from catalogs import CATALOGS`，按任务取候选目录。
"""

from catalogs import CATALOGS

_SOFTMAX = {item["name"]: item["source"] for item in CATALOGS["softmax"]}

BASELINE = ""
BROKEN_SYNTAX = _SOFTMAX["broken_syntax"]
WRONG_AXIS = _SOFTMAX["wrong_axis"]
UNSTABLE = _SOFTMAX["unstable"]
STABLE_VEC = _SOFTMAX["stable_vec"]
REDUNDANT = _SOFTMAX["redundant"]
TRITON = _SOFTMAX["triton_softmax"]
CATALOG = _SOFTMAX
