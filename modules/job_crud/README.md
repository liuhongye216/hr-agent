# 岗位版本与生命周期能力

实现位于：

- `job_csv_agent/governance.py`：版本、生命周期、审计、PostgreSQL/SQLite 控制面；
- `job_csv_agent/governed_repository.py`：租户范围和 CSV 查询投影；
- `job_csv_agent/profiles.py`：岗位字段与 `JobProfile` 转换。

删除是控制面软删除；CSV 投影中移除当前行，但不可变岗位版本、审计和归属仍保留。发布、关闭和修改均检查企业、角色与版本。
