# 自然语言招聘岗位 CSV Agent（MVP）

这个模块用 DeepSeek 做意图识别和字段抽取，用 LangGraph 管理多轮状态；只有确定性的
`CsvJobRepository` 能通过 Pandas 修改 `company_agent/data/business/jobs.csv`。新增、修改和删除
都先生成预览，只有用户显式确认后才写入。每次写入后自动重建现有 `jd_text2sql` 的 SQLite 查询库。

## 状态和必填字段

- `IDLE`：识别新建、修改、删除、查询。
- `CREATING`：累计槽位并追问。最低条件为公司名称、岗位名称，以及任职要求/岗位职责中至少一项有效岗位内容；不再为了满足必填项伪造任职要求。
- `EDITING`：模糊检索公司/岗位；多条结果先让用户选择并锁定 `job_id`。
- `CONFIRMING`：显示新增字段或修改前后值，等待确认、取消或继续修改。

LLM 只能输出 Pydantic 校验的 `StructuredCommand`，不能生成或执行 CSV/SQL 写操作。
`job_id` 使用 UUID；CSV 写入使用 sidecar 文件锁和同目录临时文件原子替换。

岗位内容先进入带证据的内部 `SemanticFact`：区分职责、任职要求、技能、福利和待澄清信息，并记录重要性、来源和是否需要确认。只有用户明确提供或已确认且分类明确的事实会投影到现有 `*_json` 字符串数组列；推断建议只保留在会话草稿和确认预览中。Repository 在格式校验之外还会阻止明显错分及未经确认的推断写入。

## 安装与配置

```powershell
cd modules\job_csv_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item ..\..\.env.example ..\..\.env
# 编辑仓库根目录 .env，把 replace-me 替换为有效的 DEEPSEEK_API_KEY
```

## Chainlit 前端

先按下一节启动 FastAPI（端口 `8000`），再另开终端运行 Chainlit（端口 `8001`）：

```powershell
$env:JOB_AGENT_API_URL="http://127.0.0.1:8000"
chainlit run job_csv_agent/chainlit_app.py -w --port 8001
```

也可以直接运行已提供的启动脚本：

```powershell
.\run_chainlit.ps1
```

浏览器访问 `http://localhost:8001`。`JOB_AGENT_API_URL` 指向后端监听地址，不能把
Chainlit 自己也绑定到同一个端口。

Chainlit 只通过 HTTP 调用 FastAPI，不直接接触 CSV。确认卡片会显示“确认写入 / 继续修改 / 取消”按钮；模糊检索有多个结果时会显示岗位选择按钮。

## FastAPI 后端

```powershell
uvicorn job_csv_agent.api:app --reload --port 8000
```

启动后可先访问 `http://127.0.0.1:8000/health`；`llm_configured` 必须为 `true`。

最小 API 流程：

1. `POST /sessions` 创建会话。
2. `POST /sessions/{id}/messages`，body 为 `{"content":"我想招一个 Python 开发..."}`。
3. 按返回的缺失字段继续发消息。
4. `POST /sessions/{id}/confirm` 执行待确认写入；也可调用 `/cancel` 或 `/select/{index}`。

MVP 的 FastAPI 会话状态保存在单进程内存中。多 worker 或生产部署时，将
`InMemorySessionStore` 替换为 Redis，并保留同一会话的串行更新语义。

## 测试

```powershell
python -m pytest -q
```

当前模块测试覆盖 CRUD、预览确认、API、CSV/SQLite 同步、职责/要求语义拆分、职责-only 草稿，以及 inferred 信息确认前后的 Repository 写入边界。
