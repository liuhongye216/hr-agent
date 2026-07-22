# HR Agent：企业侧岗位画像与发布工作台

这是企业侧招聘 Agent 的第一版实现。当前范围是：

```text
企业输入 JD / 修改要求
→ 画像抽取与证据校验
→ 单问题追问
→ 差异预览与企业确认
→ 保存不可变岗位版本
→ 质量门审查
→ 审核通过后发布 / 关闭
```

推荐匹配、校招地图、广播文案、猎头服务和 A2A 沟通不在本版范围内。

## 实际架构

```text
Chainlit / FastAPI
        ↓
签名身份、企业租户、操作者角色
        ↓
EnterpriseHrAgent 显式状态图
  authorize
    → select_skills
    → route_lifecycle
    → invoke_job_capability
    → attach_quality_status
        ↓
JobCsvAgent（岗位编辑 Capability）
  LLM 类型化理解
    → 原文证据与 Patch 校验
    → 追问预算
    → 预览 / 明确确认
        ↓
GovernedJobRepository
  PostgreSQL 或本地 SQLite：租户、版本、状态、审计、会话、工作流指标
  jobs.csv：兼容 Text2SQL 的可重建查询投影
```

状态分成两类：

- 对话状态：`IDLE / COLLECTING / WAITING_ANSWER / WAITING_CONFIRMATION / COMPLETED / CANCELLED`；
- 岗位生命周期：`DRAFT / NEEDS_CLARIFICATION / REVIEWED / PUBLISHED / CLOSED / EXPIRED / DELETED`。

## 已实现

- 版本化 `JobProfile / Evidence / JobPatch / Clarification / Review / Audit` contracts；
- 部门、岗位类别、招聘人数、专业、毕业届别、招聘批次、截止时间和多地点画像字段；
- 服务端静态 Skill Pack，带版本、允许能力和写权限声明；
- 确定性质量门与至多一次可选 Advisory Critic；
- 岗位版本、发布状态、软删除、历史与审计；
- 企业租户隔离、角色权限、HMAC 签名身份令牌；
- SQLite/PostgreSQL 持久会话和乐观并发；
- 显式节点图、节点轨迹、耗时和错误指标；
- 原有自然语言 CRUD、查询、只读 Text2SQL、证据校验和确认机制；
- 离线岗位质量评测集。

LLM 只能产生类型化解释和增量 Patch，不能直接保存、删除、发布、执行 DML 或绕过确认。

## 本地启动

```powershell
cd modules\job_csv_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item ..\..\.env.example ..\..\.env
uvicorn job_csv_agent.api:app --reload --port 8000
```

另开终端运行 Chainlit：

```powershell
cd modules\job_csv_agent
.\.venv\Scripts\Activate.ps1
.\run_chainlit.ps1
```

本地默认允许演示身份。正式环境必须设置：

```text
HR_AGENT_ALLOW_DEMO_IDENTITY=false
HR_AGENT_AUTH_SECRET=<高强度随机密钥>
HR_AGENT_DATABASE_URL=postgresql://...
```

签名令牌负载包含 `company_id`、`operator_id`、`roles` 和 `exp`；企业和操作者身份不会接受请求头覆盖。

## 主要 API

- `POST /sessions`
- `POST /sessions/{session_id}/messages`
- `POST /sessions/{session_id}/confirm`
- `POST /sessions/{session_id}/publish`
- `POST /sessions/{session_id}/close`
- `POST /sessions/{session_id}/cancel`
- `GET /audit`
- `GET /metrics`
- `GET /health`

发布要求当前岗位版本通过质量门，且操作者具有 `hr_admin` 或 `company_admin` 角色。

## 数据后端

- 未设置 `HR_AGENT_DATABASE_URL`：使用 `runtime/hr_agent.sqlite`；
- 设置 PostgreSQL URL：岗位控制面和会话使用 PostgreSQL；
- `data/business/jobs.csv` 保留为兼容投影，Text2SQL SQLite 可随时重建，不是版本和审计真源。

数据库初始 DDL 位于 `infra/postgres/001_hr_agent_control_plane.sql`。

## 测试与评测

```powershell
$env:PYTHONPATH = "modules\job_csv_agent;modules\jd_text2sql"
New-Item -ItemType Directory -Force .pytest_tmp | Out-Null
python -m pytest -q modules\job_csv_agent\tests modules\jd_text2sql\tests --basetemp=.pytest_tmp\full
python evaluation\run_profile_quality.py
```

当前自动化结果：117 项通过，2 项需要真实外部模型而跳过。

## 目录

```text
contracts/                              共享协议说明
skills/                                 服务端静态 Skill Pack
evaluation/                             离线质量评测
infra/postgres/                         PostgreSQL DDL
modules/job_csv_agent/hr_agent_contracts/ 版本化 Python contracts
modules/job_csv_agent/job_csv_agent/    图编排、能力、治理、API
modules/jd_text2sql/                     数据准备与只读 Text2SQL
runtime/                                本地 SQLite（不提交）
```
