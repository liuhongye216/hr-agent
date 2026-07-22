# 共享数据协议

正式 Python contracts 位于 `modules/job_csv_agent/hr_agent_contracts`，当前 schema 版本为 `1.0.0`。

已定义：

- `ActorContext`
- `JobProfile`
- `FieldFact` / `Evidence`
- `JobPatch`
- `ClarificationQuestion` / `ClarificationAnswer`
- `JobProfileReview` / `ReviewIssue`
- `JobProfileVersion`
- `AuditEvent`
- `ConversationPhase` / `JobLifecycleStatus`

数据库、CSV 和模型输出不是跨模块协议；其他模块只能依赖这些版本化 contracts 或正式 API。
