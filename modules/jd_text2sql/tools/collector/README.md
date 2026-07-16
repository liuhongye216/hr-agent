# BOSS JD Collector

## 快速开始

在 PowerShell 中进入本目录后运行：

```powershell
.\run_collector.ps1 -Count 100
```

默认数据写入本工具目录的 `data/`，保持采集器可以独立打包运行。若要写入其他临时目录，可使用 `-OutputDir`。采集完成并验收后，只把 `jobs.jsonl` 更新到 Agent 根目录的 `company_agent/data/source/jobs.jsonl`；随后由 `python -m jd_text2sql export-source` 统一生成源 CSV。逐条 `raw/` 文件只用于断点续采，不进入正式数据目录。

首次运行会在本目录创建 `.venv` 并安装依赖，然后启动一个可见的 Edge/Chrome 专用会话。请在该窗口自行登录；若 BOSS 显示安全验证，请手动完成，再回到 PowerShell 按 Enter。脚本不会自动填写账号或绕过验证。

## 断点续采

若站点再次要求安全验证，采集器会停止且保留已经完成的记录。完成验证后再次运行同一命令即可继续。

## 输出

- `data/raw/<job_id>.json`：逐条保存，支持断点续采。
- `data/jobs.jsonl`：聚合后的完整数据；正文来自登录会话下的服务端 HTML。
- `data/jobs.csv`：适合 Excel 查看的一组核心字段。
- `data/errors.jsonl`：失败明细。
- `data/collection_report.json`：目标数量与完成状态。

## 约束

每条记录含 `jd_char_count`、`jd_source_mode` 和 `jd_is_complete`，若检测到登录受限正文或安全页会停止而不是写入残缺数据。

采集器单线程运行，默认每条间隔 2～4 秒；不绕过验证码，不读取浏览器隐私数据，不自动登录、不投递简历、不联系招聘者。仅应用于你有权进行的内部研究，并遵守网站协议与适用法律。
