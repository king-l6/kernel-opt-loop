"""兼容旧入口：原来的 StandIn 已拆成 proposers.ScriptedProposer / LLMProposer。

新代码请直接 `from proposers import make_proposer`。
"""

from proposers import ScriptedProposer

StandIn = ScriptedProposer
