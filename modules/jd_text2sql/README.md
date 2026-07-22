# JD → CSV → Text2SQL

这是公司侧 Agent 中的轻量 JD 数据准备与查询模块。数据不存放在模块内部，统一归属 `hr-agent/data`。

## 唯一数据流

```text
hr-agent/data/source/jobs.jsonl
        │ 纯规则：展开 JSON 顶层字段，嵌套值编码为 JSON
        ▼
hr-agent/data/source/jobs.csv
        │ 规则处理标量；DeepSeek 可选补充长文本抽取
        ▼
hr-agent/data/business/jobs.csv
        │ 一张业务表加载到可重建 SQLite
        ▼
hr-agent/runtime/jd_text2sql.sqlite
```

数据层只有三份正式文件：

- `data/source/jobs.jsonl`：采集真源，不在转换阶段修改。
- `data/source/jobs.csv`：按源字段无损展开，供检查和后续处理。
- `data/business/jobs.csv`：一岗一行的必要业务字段，也是数据库唯一输入。

业务表中的要求、职责、技能、证书和福利使用 JSON 数组列保存，因此不再维护维表、关系表、版本表、修正表或运行时 facts JSONL。

## 两阶段准备

```powershell
cd hr-agent\modules\jd_text2sql
python -m pip install -r requirements-dev.txt

# 第一步：JSONL → 源 CSV，始终是确定性规则
python -m jd_text2sql export-source

# 第二步：源 CSV → 单一业务 CSV
# 本地规则模式适合测试、离线回退
python -m jd_text2sql build-business --mode rules

# 正式混合模式：规则负责标量，LLM 只补充 jd_raw 长文本
python -m jd_text2sql build-business --mode hybrid --allow-external-llm
```

也可以一次运行两个阶段：

```powershell
python -m jd_text2sql prepare --mode hybrid --allow-external-llm
```

混合模式会把 `title` 和 `jd_raw` 发给 DeepSeek；没有显式的 `--allow-external-llm` 就会拒绝执行。LLM 输出中的每项信息必须携带能在 `jd_raw` 中逐字找到的证据，业务 CSV 仅保存校验后的简短值。

## 查询

```powershell
python -m jd_text2sql build-db
python -m jd_text2sql ask "北京有多少本科及以上岗位" --mode rules
python -m jd_text2sql ask "需要 Excel 技能的岗位" --mode hybrid
python -m pytest -q
```

SQLite 只包含 `jobs` 一张业务表和内部元数据表。`requirements_json`、`responsibilities_json`、`skills_json`、`certificates_json`、`benefits_json` 可通过 SQLite `json_each` 查询。

## DeepSeek 配置

在 `hr-agent/.env`（或模块本地 `.env`）中配置：

```dotenv
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```
