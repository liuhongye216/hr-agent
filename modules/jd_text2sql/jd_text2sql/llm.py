from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, EXTRACTION_MODEL, TEXT2SQL_MODEL


class SemanticEvidenceFact(BaseModel):
    value: str
    category: Literal["responsibility", "requirement", "skill", "benefit", "unknown"]
    importance: Literal["must", "preferred", "neutral", "unknown"]
    source_type: Literal["explicit", "inferred", "user_confirmed"]
    evidence_text: str
    needs_confirmation: bool = False
    source_section: Literal[
        "job_description", "responsibilities", "work_content", "requirements",
        "qualifications", "benefits", "unknown",
    ] = "unknown"
    source_item_index: int | None = None
    subject: Literal["employee", "candidate", "company", "unknown"] = "unknown"
    action: str | None = None
    action_object: str | None = None


class LLMLongTextExtraction(BaseModel):
    facts: list[SemanticEvidenceFact] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)


class ExtractedValues(BaseModel):
    requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certificates: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)


class LLMSQLDraft(BaseModel):
    sql: str
    parameters_json: str = "[]"
    explanation: str
    needs_clarification: bool = False
    clarification_question: str | None = None

    def parameters(self) -> list[Any]:
        value = json.loads(self.parameters_json)
        if not isinstance(value, list) or any(isinstance(item, (dict, list)) for item in value):
            raise ValueError("parameters_json must be a JSON array of scalar values")
        return value


def _client():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; configure it in company_agent/.env")
    from openai import OpenAI

    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _json_completion(
    response_model: type[ModelT],
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> ModelT:
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": f"{system}\n只输出符合以下 schema 的 JSON 对象：{schema}",
        },
        {"role": "user", "content": user},
    ]
    last_error: Exception | None = None
    for _ in range(3):
        response = _client().chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            if not content:
                raise ValueError("DeepSeek returned empty JSON content")
            return response_model.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            messages.extend(
                [
                    {"role": "assistant", "content": content or "{}"},
                    {"role": "user", "content": f"JSON 无效：{exc}。请输出完整有效的 JSON。"},
                ]
            )
    raise RuntimeError(f"DeepSeek did not return valid structured JSON: {last_error}")


LONG_TEXT_SYSTEM = """你是招聘 JD 长文本语义事实抽取器。标量字段由确定性规则处理。必须先把 jd_raw 拆成每句只有一个语义的原子句，再分类：
- responsibility：员工入职后执行的动作、任务或交付结果；负责、完成、搭建、制定、推进、交付、参与等通常是职责。
- requirement：候选人入职前需具备的学历、经验、能力、知识、证书或特质；具备、熟悉、掌握、学历、经验、证书、优先等通常是资格。
- skill：原文明示的工具、技术、语言或专业技能标签。
- benefit：公司提供的五险一金、奖金、假期、补贴等福利。
- unknown：证据不足或存在歧义，必须 needs_confirmation=true，不能强行归类。

每条事实输出 value、category、importance(must/preferred/neutral/unknown)、source_type(explicit/inferred/user_confirmed)、evidence_text、needs_confirmation。evidence_text 必须是 jd_raw 中连续且逐字一致的原文。只有 explicit 且分类明确或 user_confirmed 的事实可进入业务字段；inferred 只能作为待确认建议，unknown 必须追问。禁止把常识推断包装成原文明示内容。

先识别职位描述、岗位职责、工作内容、职位要求、任职要求、任职资格和福利待遇章节，并输出 source_section、source_item_index、subject、action、action_object。章节是重要上下文，但句子真实语义优先：候选人“具备风险分析能力”是 requirement；“推进风险决策能力的建模和优化”是 responsibility；“开展 Agent 能力评测”中的能力是动作对象，不是候选人资格。“参与训练环境的设计、搭建与迭代”必须保持为一个完整动作，不能产生孤立的“搭建与迭代”。

示例：
- “从0开始预训练大模型”是 responsibility，不是 requirement。
- “具备从0开始预训练大模型的经验”是 requirement。
- “负责训练数据清洗和模型预训练”是 responsibility。
- “熟悉 Python 和 PyTorch”是 requirement，并可产生有同一原文证据的 skill 标签。
- “参与模型预训练，有分布式训练经验优先”拆成 responsibility 和 preferred requirement。
- “本科及以上学历，负责训练平台建设”拆成 must requirement 和 responsibility。
反例：不得把“负责平台建设”写为 requirement；不得从“预训练大模型”自行推断候选人已会 Python、Transformer 或分布式训练。若提出这些能力，只能标记 inferred、needs_confirmation=true，且不得投影。"""


def extract_long_text_with_llm(
    job: dict[str, Any], model: str = EXTRACTION_MODEL
) -> ExtractedValues:
    raw = str(job.get("jd_raw") or "")
    parsed = _json_completion(
        LLMLongTextExtraction,
        model=model,
        system=LONG_TEXT_SYSTEM,
        user=json.dumps({"title": job.get("title"), "jd_raw": raw}, ensure_ascii=False),
        max_tokens=8_000,
    )

    projected: dict[str, list[str]] = {
        "requirement": [], "responsibility": [], "skill": [], "benefit": [],
    }
    for fact in parsed.facts:
        value = fact.value.strip()
        if (
            value
            and fact.evidence_text in raw
            and fact.category in projected
            and (
                fact.source_type == "user_confirmed"
                or (fact.source_type == "explicit" and not fact.needs_confirmation)
            )
        ):
            if value not in projected[fact.category]:
                projected[fact.category].append(value)

    # Evidence is used as a guard at generation time; the lightweight business table stores values only.
    return ExtractedValues(
        requirements=projected["requirement"],
        responsibilities=projected["responsibility"],
        skills=projected["skill"],
        certificates=[
            fact.value.strip() for fact in parsed.facts
            if fact.category == "requirement"
            and (
                fact.source_type == "user_confirmed"
                or (fact.source_type == "explicit" and not fact.needs_confirmation)
            )
            and fact.evidence_text in raw
            and any(marker in fact.value for marker in ("证", "执照", "资质"))
        ],
        benefits=projected["benefit"],
    )


SCHEMA_CONTEXT = """只允许查询 SQLite 表 jobs，一岗一行。主要字段：
job_id, title, company_name, city, work_address,
salary_min, salary_max, salary_currency, salary_period,
recruitment, employment, work_mode, education_min_level,
experience_min_months, experience_max_months,
requirements_json, responsibilities_json, skills_json, certificates_json, benefits_json,
source_url, scraped_at。
五个 *_json 字段都是 JSON 字符串数组；检索数组内容时使用 json_each，例如：
EXISTS (SELECT 1 FROM json_each(j.skills_json) x WHERE x.value = ?)。"""


def text2sql_with_llm(question: str, model: str = TEXT2SQL_MODEL) -> LLMSQLDraft:
    system = f"""你是只读 SQLite Text2SQL 生成器。
{SCHEMA_CONTEXT}

只生成一条 SELECT 或 WITH...SELECT，不得输出注释或分号。用户值使用 ? 占位符，
并按顺序放入 parameters_json。薪资比较必须同时限定 salary_currency 和 salary_period。
非聚合查询最多返回 100 行；不确定时设置 needs_clarification=true。"""
    return _json_completion(
        LLMSQLDraft,
        model=model,
        system=system,
        user=question,
        max_tokens=4_000,
    )
