# 工作流编排

编排已实现于 `modules/job_csv_agent/job_csv_agent/workflow.py` 和 `enterprise_agent.py`。

当前显式图：

```text
authorize
→ select_skills
→ route_lifecycle
   ├─ publish / close / history → END
   └─ 普通岗位对话
      → invoke_job_capability
      → attach_quality_status
      → END
```

每次运行限制最大节点数并保存节点轨迹、耗时、完成状态和错误类型。人工确认仍由独立 API 和会话版本控制，不由模型输出触发。

原 `JobCsvAgent.handle()` 现在是岗位编辑 Capability 内部控制器；跨身份、Skill、生命周期、质量门和观测的编排全部位于企业 Agent 图中。
