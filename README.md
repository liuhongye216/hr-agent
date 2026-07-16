# HR Agent：公司侧招聘岗位 Agent

这是一个可运行的公司侧招聘 MVP。当前主线聚焦两件事：把自然语言岗位信息安全地写入单一业务 CSV，以及基于该 CSV 提供受保护的只读 Text2SQL 查询。

## 当前状态

| 能力 | 状态 | 说明 |
|---|---|---|
| JD JSONL/CSV 数据准备 | 已实现 | 支持规则模式与需显式授权的 DeepSeek 混合抽取 |
| 职责/要求语义识别 | 已实现 | 原子句拆分、证据与来源标记、复合句重分类、待确认推断建议 |
| 自然语言岗位新增/修改/删除 | 已实现（MVP） | LangGraph 多轮状态、候选消歧、预览和明确确认后写入 |
| CSV Repository | 已实现 | Pydantic 格式校验、独立语义闸门、文件锁、原子替换、内容哈希 |
| 岗位查询 | 已实现（MVP） | 规则查询优先，LLM 回退；SQL 仅允许受限的只读查询 |
| FastAPI / Chainlit | 已实现（本地单进程） | Chainlit 只调用 API，不直接读写 CSV |
| 采集工具 | 已实现（辅助工具） | 登录后单线程采集、断点续采，不绕过验证 |
| 独立追问服务 | 部分实现 | 高价值追问已在 `job_csv_agent` 内实现；独立模块仍是边界占位 |
| 版本化 JobPatch、软删除、审计日志 | 未实现 | 当前删除为物理删除，CSV 不保存岗位版本 |
| 候选人检索与统一匹配 | 未实现 | `retrieval` 与 `matching_gateway` 目前只有设计边界 |
| 生产认证、权限、持久会话 | 未实现 | 当前无 RBAC；API 会话保存在单进程内存中 |

详细盘点与后续建议见 [docs/STATUS.md](docs/STATUS.md)。

## 核心数据流

```text
本地 data/source/jobs.jsonl
  → jd_text2sql：规则/混合抽取
  → data/business/jobs.csv
  → job_csv_agent：自然语言 CRUD、语义校验、预览确认
  → runtime/jd_text2sql.sqlite（可重建，不提交）
  → 规则优先 + LLM 回退的只读查询
```

业务 CSV 继续使用兼容的 JSON 字符串数组列。会话内部使用带 `category / importance / source_type / evidence_text / needs_confirmation` 的语义事实；只有明确事实或用户已确认事实能够投影到正式字段。

## 快速开始

项目使用模块级依赖文件。建议使用 Python 3.11 或 3.12。

```powershell
git clone https://github.com/liuhongye216/hr-agent.git
cd hr-agent
Copy-Item .env.example .env
# 编辑 .env，填入有效的 DEEPSEEK_API_KEY

cd modules\job_csv_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn job_csv_agent.api:app --reload --port 8000
```

另开终端启动 Chainlit：

```powershell
cd modules\job_csv_agent
.\.venv\Scripts\Activate.ps1
.\run_chainlit.ps1
```

浏览器访问 `http://localhost:8001`。后端健康检查为 `http://127.0.0.1:8000/health`。

## JD 数据与 Text2SQL

```powershell
cd modules\jd_text2sql
python -m pip install -r requirements-dev.txt
python -m jd_text2sql build-db
python -m jd_text2sql ask "北京有多少本科及以上岗位" --mode rules
```

若要从自己的 JD 真源重新生成业务 CSV，请先把有权处理的数据放到本地 `data/source/jobs.jsonl`，再参考 [modules/jd_text2sql/README.md](modules/jd_text2sql/README.md)。混合抽取会向外部模型发送 `title` 和 `jd_raw`，必须显式传入 `--allow-external-llm`。

## 测试

从仓库根目录运行完整测试：

```powershell
$env:PYTHONPATH = "modules\job_csv_agent;modules\jd_text2sql"
python -m pytest -q modules\job_csv_agent\tests modules\jd_text2sql\tests
```

当前完整测试共 33 项，覆盖语义分类、职责-only 草稿、推断确认边界、CRUD、确认流程、CSV 同步、Text2SQL 与 SQL 安全闸门。

## 目录

```text
data/business/             业务 jobs.csv
data/source/               本地采集真源（默认不提交）
contracts/                 共享协议规划与当前语义事实约定
docs/                      状态与设计记录
modules/jd_text2sql/       数据准备、SQLite、只读 Text2SQL
modules/job_csv_agent/     自然语言 CRUD、FastAPI、Chainlit
modules/clarification/     独立追问模块占位
modules/job_crud/          版本化岗位写入模块占位
modules/retrieval/         候选人召回占位
modules/matching_gateway/  匹配服务网关占位
interfaces/                未来 HTTP/A2A 边界
orchestration/             未来跨模块编排边界
```

## 数据与安全边界

- `.env`、虚拟环境、运行时 SQLite、采集器浏览器 profile 与原始采集数据均不提交。
- `data/business/jobs.csv` 是演示与查询用业务表；公开分发前仍应确认数据使用授权与站点条款。
- LLM 不执行 CSV 或 SQL 写操作；所有岗位写入由确定性 Repository 完成，并要求用户明确确认。
- 当前是本地 MVP，不应直接暴露到公网；生产使用前必须补认证、授权、持久会话、审计与备份。
