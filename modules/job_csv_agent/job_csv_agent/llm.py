from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, is_valid_api_key
from .schemas import StructuredCommand
from .semantics import reconcile_command_semantics


SYSTEM_PROMPT = """你是公司侧招聘岗位管理助手，只把用户文本转换为结构化命令，绝不执行 CSV/SQL/文件操作。
intent 只能是 create/update/delete/search/confirm/cancel/unknown。处理岗位内容时必须先拆成每句只含一个语义的原子句，再逐句生成 semantic_facts，最后投影业务字段。

语义定义：
- responsibility（岗位职责）：员工入职后要执行的动作、承担的任务或交付的结果。常见动词：负责、完成、搭建、制定、推进、交付、参与、建设、训练、评估。
- requirement（任职要求）：候选人入职前要具备的学历、经验、能力、知识、证书或行为特质。常见表达：具备、熟悉、掌握、学历、经验、证书、优先。
- skill（技能）：用户明确提到的工具、技术、语言或专业技能标签；技能标签可与一条 requirement 共用 evidence_text，但不能凭常识补全。
- benefit（福利）：公司向员工提供的薪酬之外待遇，如五险一金、年终奖、带薪年假。
- unknown（待澄清）：原文不足以可靠分类或存在关键歧义；必须 needs_confirmation=true，不能强塞入业务字段。

每条 semantic_fact 必须包含 value、category、importance(must/preferred/neutral/unknown)、source_type(explicit/inferred/user_confirmed)、evidence_text、needs_confirmation。
- explicit：用户原文明确事实；evidence_text 必须对应原文。
- inferred：根据职责推测的候选能力，只能作为建议且 needs_confirmation=true。
- user_confirmed：用户在当前上下文中明确确认过的建议。
只有 explicit 且 category 明确，或 user_confirmed 的事实，才可投影到 requirements_json、responsibilities_json、skills_json、benefits_json。inferred 和 unknown 一律不得投影。不得因为旧流程曾要求 requirements_json 非空而伪造任职要求，也不得把常识推断说成用户明确提供。

正例、反例与复合句拆分：
- “从0开始预训练大模型” -> responsibility/neutral/explicit；不是 requirement。
- “具备从0开始预训练大模型的经验” -> requirement/must/explicit；不是 responsibility。
- “负责训练数据清洗和模型预训练” -> responsibility/neutral/explicit。
- “熟悉 Python 和 PyTorch” -> 一条 requirement/must/explicit；还可生成 Python、PyTorch 两条 skill/explicit 标签。
- “参与模型预训练，有分布式训练经验优先” -> 拆成 responsibility“参与模型预训练”和 requirement/preferred“有分布式训练经验优先”。不得保留成一条。
- “本科及以上学历，负责训练平台建设” -> 拆成 requirement/must“本科及以上学历”和 responsibility/neutral“负责训练平台建设”。
- “提供五险一金，参与推荐系统开发” -> 拆成 benefit 与 responsibility。
- 反例：“从0开始预训练大模型”不能改写成“具备大模型预训练经验”；“负责平台建设”不能写入 requirements_json；“熟悉 Python”不能写入 responsibilities_json。

当只明确给出“从0开始预训练大模型”一类职责时，职责保持 explicit；可以给出 Python、深度学习框架、Transformer、数据处理、分布式训练等少量 inferred 技能建议，并生成高信息量 clarification_questions，询问“从0”含义、负责环节、技能必需性及是否要求既有大规模/分布式训练经验。建议绝不进入正式 JSON 字段。

普通字段约定：title 岗位名；company_name 公司名；city 城市；work_address 详细地址；salary_min/salary_max 换算成人民币数值（20k-30k/月 => 20000,30000,CNY,month）；education_min_level 不限=0、初中=1、高中/中专=2、大专=3、本科=4、硕士=5、博士=6；experience_*_months 使用月。用户明确清空字段时写 clear_fields。修改/删除的定位文字写 search_query，仅把真正变更写 fields。不得生成 job_id、content_hash、scraped_at、extraction_mode。未明确的信息保持 null 或空数组。"""


class LLMConfigurationError(RuntimeError):
    pass


class LLMServiceError(RuntimeError):
    pass


class StructuredInterpreter:
    def __init__(self, model: str = DEEPSEEK_MODEL) -> None:
        self.model = model

    def extract(self, text: str, context: dict[str, Any]) -> StructuredCommand:
        if not is_valid_api_key(DEEPSEEK_API_KEY):
            raise LLMConfigurationError(
                "DEEPSEEK_API_KEY 未配置或仍是示例占位值。请在 company_agent/.env 中填写有效密钥，"
                "然后重启 FastAPI。"
            )
        from openai import (
            APIConnectionError, APIStatusError, AuthenticationError, BadRequestError,
            OpenAI, RateLimitError,
        )

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        schema = json.dumps(StructuredCommand.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n只输出符合此 schema 的 JSON：{schema}"},
            {"role": "user", "content": json.dumps({"context": context, "input": text}, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=3_000,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except AuthenticationError as exc:
                raise LLMConfigurationError(
                    "DeepSeek 鉴权失败，请检查 company_agent/.env 中的 DEEPSEEK_API_KEY，"
                    "更新后重启 FastAPI。"
                ) from exc
            except BadRequestError as exc:
                raise LLMServiceError(
                    f"DeepSeek 拒绝了请求，请检查模型名 DEEPSEEK_MODEL={self.model} 是否可用。"
                ) from exc
            except RateLimitError as exc:
                raise LLMServiceError("DeepSeek 当前限流或额度不足，请稍后重试。") from exc
            except APIConnectionError as exc:
                raise LLMServiceError("无法连接 DeepSeek API，请检查网络和 DEEPSEEK_BASE_URL。") from exc
            except APIStatusError as exc:
                raise LLMServiceError(f"DeepSeek API 返回异常状态码 {exc.status_code}。") from exc
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                command = StructuredCommand.model_validate_json(content)
                return reconcile_command_semantics(command, text)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                messages.extend([
                    {"role": "assistant", "content": content or "{}"},
                    {"role": "user", "content": f"JSON 不符合 schema：{exc}。请重新输出。"},
                ])
        raise RuntimeError(f"DeepSeek 未返回有效结构化 JSON：{last_error}")
