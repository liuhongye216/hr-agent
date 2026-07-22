# 企业 Agent 接口

FastAPI 实现在 `modules/job_csv_agent/job_csv_agent/api.py`。

写操作使用两层并发版本：

- `expected_version`：会话状态版本；
- `expected_job_version`：岗位画像版本，发布和更新时防止覆盖其他操作者修改。

身份由 `IdentityProvider` 提供。配置 `HR_AGENT_AUTH_SECRET` 后只接受 HMAC 签名 Bearer 令牌中的企业、操作者和角色，拒绝请求头覆盖。

主要接口：会话创建、岗位对话、确认、取消、候选岗位选择、发布、关闭、审计和指标。推荐匹配、校招地图、广播、猎头和 A2A 接口不在本版定义。
