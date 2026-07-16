# 共享数据协议（占位）

这里将保存模块间稳定的数据模型，而不是数据库实现细节。优先定义：

- `JobProfile`：标准岗位画像；
- `Evidence`：字段来源和原文证据；
- `ClarificationQuestion` / `ClarificationAnswer`；
- `JobPatch`：字段级增删改及修改原因；
- `RetrievalRequest` / `CandidateSummary`；
- `MatchRequest` / `MatchResult`；
- `AgentState`：编排状态。

所有协议都需要 `schema_version`。岗位事实应区分 `must`、`preferred`、`neutral` 和 `unknown`，并保留来源、是否确认和是否需要追问。

当前 `job_csv_agent` 的会话内语义事实采用以下最小结构（业务 CSV 继续保存兼容的 JSON 字符串数组，不直接增加这些列）：

- `value`：忠实、原子化的事实值；
- `category`：`responsibility / requirement / skill / benefit / unknown`；
- `importance`：`must / preferred / neutral / unknown`；
- `source_type`：`explicit / inferred / user_confirmed`；
- `evidence_text`：支持该事实的用户原文；
- `needs_confirmation`：是否仍需用户确认。

正式字段只接收分类明确的 `explicit` 或 `user_confirmed` 事实。`inferred` 与 `unknown` 留在会话草稿，用于建议和追问，不得直接写入业务 CSV。
