# 自然语言招聘岗位 CSV Agent（MVP）

这个模块用 DeepSeek 做意图识别和字段抽取，用 LangGraph 管理多轮状态；只有确定性的
`CsvJobRepository` 能通过 Pandas 修改 `company_agent/data/business/jobs.csv`。新增、修改和删除
都先生成预览，只有用户显式确认后才写入。每次写入后自动重建现有 `jd_text2sql` 的 SQLite 查询库。

## 状态和必填字段

- `IDLE`：识别新建、修改、删除、查询。
- `CREATING`：累计槽位并追问。最低条件为公司名称、岗位名称，以及任职要求/岗位职责中至少一项有效岗位内容；不再为了满足必填项伪造任职要求。
- `EDITING`：模糊检索公司/岗位；多条结果先让用户选择并锁定 `job_id`。
- `CONFIRMING`：显示新增字段或修改前后值，等待确认、取消或继续修改。

所有开放式话语都交给 LLM 理解，包括意图、动作、岗位字段、创建态中的“补充 / 闲聊 / 帮助 / 查询 / 新任务”区分，以及 JD 章节、职责/要求分类和受约束改写。业务路径不再运行关键词路由、岗位名前后缀截取或正则语义修复。“我要招聘”由模型输出创建动作且岗位名为空；“我要招聘算法工程师”由模型输出完整岗位名。确定性快速路径只处理确认、取消和候选序号。

LLM 只能输出 Pydantic 校验的严格 `LLMInterpretation`：包含路由、动作、字段、字段证据、语义事实、整体/路由置信度、待澄清项、JD 覆盖和自然回复，不能生成或执行 CSV/SQL 写操作。低置信度或 `requires_clarification=true` 时只追问，不合并字段；模型配置、网络或协议不可用时保留原草稿并明确说明没有修改字段。
`job_id` 使用 UUID；CSV 写入使用 sidecar 文件锁和同目录临时文件原子替换。

岗位内容先进入带证据的内部 `SemanticFact`：区分职责、任职要求、技能、福利和待澄清信息，并记录重要性、来源和是否需要确认。只有用户明确提供或已确认且分类明确的事实会投影到现有 `*_json` 字符串数组列；推断建议只保留在会话草稿和确认预览中。Repository 只执行类型、必填身份字段、JSON、数值范围、未确认推断和事实/目标字段显式冲突等数据完整性硬校验，不会在用户确认保存后用低置信度关键词规则再次否决。

确定性后处理不再判断文本属于职责还是要求，也不再从句子中抽取经验、学历、技能或用工类型。它只做 Pydantic 字段格式、空白/句末标点、精确去重、原文证据存在性、事实类别到目标字段的一致性和未确认推断隔离。JD 的原子拆分、章节识别、分类与忠实改写均由模型完成；每条结果必须带原文证据，任何缺证据、推断或待澄清事实都不会进入正式字段。

预览使用严格有效的中文业务 JSON 代码块，`null` 表示未填写，列表字段始终为 JSON 数组。用户可复制该 JSON、直接编辑并发回；系统只做字段格式校验，不对 JSON 内容进行关键词语义重分类，也不会直接写入 CSV。空数组表示明确清空列表；格式错误会返回行列位置并保留原草稿。

语义检查只验证模型声明的分类、投影字段和证据边界：未确认推断、待澄清事实、缺失原文证据或类别与目标字段冲突会阻止写入。普通界面只显示中文说明，不暴露内部状态名。

最低创建条件仍是公司名称、岗位名称，以及岗位职责或任职要求至少一项。达到最低条件后即可预览；城市、办公模式、用工类型、薪资、学历、经验、技能及实习出勤等重要信息若缺失，用户可以跳过，但首次选择按当前内容保存时会收到一次完整提醒，只有再次明确确认后才写入。提醒确认后不会循环阻止保存。

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

Chainlit 只通过 HTTP 调用 FastAPI，不直接接触数据文件。确认卡片会显示“按当前内容保存 / 继续补充 / 取消”按钮；模糊检索有多个结果时会显示岗位选择按钮。普通聊天消息不会显示内部编号、存储字段名或实现状态名。

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

当前模块测试覆盖 LLM 类型化协议、“我要招聘”空岗位名、完整岗位名抽取、创建态闲聊隔离、低置信度追问、模型不可用降级、CRUD 预览/确认/取消、中文 JSON、Repository 文件写入、查询库同步、COUNT 和 SQL Guard。测试使用 fake interpreter，不调用真实外部模型。
