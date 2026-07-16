# 工作流编排（占位）

该目录将实现公司侧 Agent 的确定性状态机，不在这里重复实现抽取、写库或检索逻辑。

建议状态：

```text
DRAFT
→ EXTRACTED
→ NEEDS_CLARIFICATION
→ PENDING_CONFIRMATION
→ ACTIVE
→ CLOSED
```

建议主流程：

```text
接收输入
→ 意图识别
→ JD 抽取/读取现有岗位
→ 完整性与冲突检测
→ 追问循环
→ 生成字段 Patch
→ 差异预览和确认
→ 保存岗位版本
→ 发布检索请求
→ 返回匹配解释
```

编排状态至少应记录 `session_id`、`company_id`、`operator_id`、`intent`、`job_id`、`job_profile_version`、缺失字段、冲突、待问问题、待确认 Patch 和工具执行结果。
