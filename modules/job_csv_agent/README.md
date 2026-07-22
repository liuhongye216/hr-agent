# 企业岗位 Agent Runtime

该模块包含两个层次：

- `EnterpriseHrAgent`：身份、Skill、生命周期、质量门和观测的显式图编排；
- `JobCsvAgent`：自然语言岗位编辑、查询、追问、预览和确认 Capability。

模型输出必须通过 `LLMInterpretation`、原文证据、字段规范化、语义冲突和 Patch 应用检查。模型不能保存、发布、删除、执行 SQL 或授予权限。

控制面使用 PostgreSQL 或 SQLite 保存企业归属、不可变岗位版本、Review、软删除、审计和持久会话；CSV 仅是兼容只读 Text2SQL 的查询投影。

## 工作流

```text
身份授权 → 静态 Skill 选择 → 生命周期路由
  ├─ 发布 / 关闭 / 历史
  └─ 岗位编辑 Capability → 质量状态 → 返回
```

岗位保存达到最低条件并通过质量门后进入 `REVIEWED`，仍需具备发布权限的操作者显式发布。

## 运行与测试

参见仓库根目录 `README.md`。测试使用注入的内存会话与临时仓储，不访问真实业务数据；默认应用运行使用持久会话和治理仓储。
