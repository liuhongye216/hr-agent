# 招聘岗位 CSV Agent

该模块使用模型理解岗位新增、补充、修改、删除和查询请求；确定性代码负责字段规范化、
patch 应用、字段覆盖校验、版本化确认、查询白名单和 CSV 原子写入。模型不能读写 CSV、
生成或执行 SQL，也不能替用户确认保存。

## 主链路

会话只保留一份正式草稿，并记录：

- `draft_revision`：草稿内容修订号；
- `preview_revision`：最近一次成功预览对应的修订号；
- `state_version`：API 并发版本；
- `pending_action`、`target_id`：当前新增/修改/删除操作；
- `can_confirm`：只有最新成功预览可确认。

每轮结构化结果包含 `intent`、`patch`、`mentioned_fields`、`evidence_spans`、
`ignored_fragments` 和 `clarification_question`。控制器先保存本轮前草稿，再规范化和应用 patch，
然后逐项检查用户明确声明的字段是否真正落入草稿。未落实时自动重试一次；仍失败则恢复本轮前
草稿、撤销确认资格，并返回“本轮字段未成功应用，尚未保存”。

`normalization.py` 集中处理中文经验年/月、招聘类型和最低学历等封闭字段。长文本职责、要求、
技能和福利仍由模型按原文分类；代码只做证据校验、Markdown 包装清理、增量 patch 和去重，
不会根据岗位名称生成默认内容。

## 查询

查询使用无 SQL 的轻量 `QueryPlan`：

- `mode`：`count`、`list` 或 `detail`；
- 过滤字段：公司、岗位、城市、招聘类型；
- `limit`：最多 10。

`CsvJobRepository` 直接在小规模 CSV 上执行包含匹配。计数只返回标量；列表只投影公司、岗位、
城市；只有明确详情请求才读取用户可见字段，并由控制器用中文标签和 Markdown 渲染。
`content_hash`、`scraped_at`、`extraction_mode` 等内部列不会进入用户消息。

## 保存与并发

新增岗位最低保存条件为公司名称、岗位名称，以及至少一条职责或任职要求。正式 CSV 只在一次
明确确认后写入，并使用文件锁、临时文件和原子替换。`POST /confirm` 必须携带最近响应的
`expected_version`；过期版本返回 409，且确认时还会核对 `preview_revision == draft_revision`。

## 运行

```powershell
cd modules\job_csv_agent
python -m pip install -r requirements.txt
uvicorn job_csv_agent.api:app --reload --port 8000
```

Chainlit 前端：

```powershell
$env:JOB_AGENT_API_URL="http://127.0.0.1:8000"
chainlit run job_csv_agent/chainlit_app.py -w --port 8001
```

## 测试

在仓库根目录运行：

```powershell
$env:PYTHONPATH="modules\job_csv_agent;modules\jd_text2sql"
python -m pytest -q modules\job_csv_agent\tests modules\jd_text2sql\tests
```

所有持久化和查询测试使用 `tmp_path` 下的临时 CSV/SQLite 文件，不读取、不写入，也不假设真实
`data/business/jobs.csv` 的记录数量。
