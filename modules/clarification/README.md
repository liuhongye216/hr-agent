# 追问能力

实现位于 `modules/job_csv_agent/job_csv_agent/clarification.py`。

它维护结构化问题、回答、跨轮问题计数和预算；岗位编辑 Capability 仍负责根据缺失、冲突和未归类片段生成当轮最高优先级问题。每轮只暴露一个问题，已回答问题不会作为新的事实自动写入，仍须经过证据与 Patch 校验。
