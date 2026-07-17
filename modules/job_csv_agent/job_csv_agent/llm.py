from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, is_valid_api_key
from .schemas import JobFields, Phase, SemanticFact, StructuredCommand
from .semantics import reconcile_command_semantics


SYSTEM_PROMPT = """你是公司侧招聘岗位管理助手，只把用户文本转换为结构化命令，绝不执行 CSV/SQL/文件操作。
intent 只能是 create/update/delete/search/confirm/cancel/unknown。处理岗位内容时必须先拆成每句只含一个语义的原子句，再逐句生成 semantic_facts，最后投影业务字段。

路由约束：包含“我要招、想招、招聘、招一个、招一名、新建岗位、发布岗位”时优先判定 create；正式工、实习生、兼职、合同工只表示用工类型。只有用户明确说修改、更新、删除、查询或查找已有岗位时，才能进入相应流程。上下文处于 CREATING 时，普通补充默认延续当前草稿，不能改判成查询或修改已有岗位。

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

逻辑关系：并且、同时、且、以及和独立谓语之间的逗号通常表示 AND，应拆成可独立阅读的原子条件，并补齐共享谓语；“或、或者、任一、至少一种、二选一、其一”表示 OR，必须保留在同一个原子条件中。“熟练掌握 Python 或 Java”不能拆成两个必须条件。对于“Spark、Flink、Doris 等工具”这类未说明全部或任选的列表，忠实保留一个概括性要求，并生成按任职要求归类的确认问题。

技能字段只能放 SQL、Python、LangGraph、FastAPI 等简短标签，不得放完整要求句。完整能力句归入任职要求，同时可从明确原文提取技能标签。列表元素不保留句末句号，不得出现“。、”“，、”“、以及”等异常标点。经验年限写入 experience_*_months 后，要求文本不再重复年限。

当只明确给出“从0开始预训练大模型”一类职责时，职责保持 explicit；可以给出 Python、深度学习框架、Transformer、数据处理、分布式训练等少量 inferred 技能建议，并生成高信息量 clarification_questions，询问“从0”含义、负责环节、技能必需性及是否要求既有大规模/分布式训练经验。建议绝不进入正式 JSON 字段。

改写约束：去掉“我要、我们想、这个岗位是”等口语前缀，把职责整理为动作表达，把模糊能力表达忠实规范化；不得新增用户未确认的学历、年限、证书、数值、技术或“熟练、精通、必须、独立负责”等强度。用户只说“懂 LLM”时可写成“具备大语言模型（LLM）相关基础知识”，不得补成 Python、PyTorch、分布式训练、SFT、RLHF 或 DPO。

普通字段约定：title 岗位名；company_name 公司名；city 城市；work_address 详细地址；salary_min/salary_max 换算成人民币数值（20k-30k/月 => 20000,30000,CNY,month）；education_min_level 不限=0、初中=1、高中/中专=2、大专=3、本科=4、硕士=5、博士=6；experience_*_months 使用月。用户明确清空字段时写 clear_fields。修改/删除的定位文字写 search_query，仅把真正变更写 fields。不得生成 job_id、content_hash、scraped_at、extraction_mode。未明确的信息保持 null 或空数组。"""


_CREATE_RE = re.compile(
    r"^(?:我要招|我想招(?:聘)?|想招聘?|帮我招|招聘|招一个|招一名|新建(?:一个)?|发布岗位)"
)
_UPDATE_RE = re.compile(r"(?:修改|更新|调整(?:一下)?(?:已有)?岗位|编辑)")
_DELETE_RE = re.compile(r"(?:删除|取消岗位|下架岗位)")
_SEARCH_RE = re.compile(r"(?:查询|查一下|查找|搜索|有哪些岗位|找一下)")
_COMPANY_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9（）()·]{2,40}?(?:有限责任公司|股份有限公司|有限公司|公司))"
)
_EMPLOYMENT = (
    (re.compile(r"(?:正式工|全职)"), "full_time"),
    (re.compile(r"(?:实习生|实习岗位|实习)"), "internship"),
    (re.compile(r"兼职"), "part_time"),
    (re.compile(r"(?:合同工|合同制)"), "contract"),
)


def _employment_fields(text: str) -> dict[str, Any]:
    for pattern, value in _EMPLOYMENT:
        if pattern.search(text):
            fields: dict[str, Any] = {"employment": value}
            if value == "internship":
                fields["recruitment"] = "internship"
            return fields
    return {}


def _create_title(text: str) -> str | None:
    candidate = re.sub(
        r"^.*?(?:我要招|我想招(?:聘)?|想招聘?|帮我招|招聘|招一个|招一名|新建(?:一个)?|发布)(?:一个|一名)?",
        "", text.strip(), count=1,
    ).strip(" ，,。.!！")
    candidate = re.sub(r"岗位$", "", candidate).strip()
    if not candidate or candidate in {"正式工", "全职", "实习生", "实习", "兼职", "合同工"}:
        return None
    return candidate


def _locating_query(text: str) -> str:
    query = re.sub(
        r"^(?:请)?(?:帮我)?(?:修改|更新|编辑|删除|查询|查一下|查找|搜索|找一下)", "", text.strip(), count=1,
    )
    return query.strip(" 的，,。.!！") or text.strip()


def deterministic_command(text: str, phase: str | Phase = Phase.IDLE.value) -> StructuredCommand | None:
    """Handle high-confidence routing/extraction before asking an LLM.

    Returning ``None`` means the text still needs general language understanding.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    employment = _employment_fields(cleaned)

    # Explicit existing-position operations take precedence, but employment words alone never do.
    if _DELETE_RE.search(cleaned):
        return StructuredCommand(intent="delete", search_query=_locating_query(cleaned))
    if _UPDATE_RE.search(cleaned):
        return StructuredCommand(intent="update", search_query=_locating_query(cleaned))
    if _SEARCH_RE.search(cleaned):
        return StructuredCommand(intent="search", search_query=_locating_query(cleaned))
    if _CREATE_RE.search(cleaned):
        title = _create_title(cleaned)
        if title:
            employment["title"] = title
        return StructuredCommand(intent="create", fields=JobFields.model_validate(employment))

    if str(phase) == Phase.CREATING.value:
        company_match = _COMPANY_RE.search(cleaned)
        fields = dict(employment)
        facts: list[SemanticFact] = []
        if company_match:
            fields["company_name"] = company_match.group(1)
            remainder = (cleaned[:company_match.start()] + " " + cleaned[company_match.end():]).strip(" ，,。.!！")
            # A short plain remainder is the position title, not a request to search existing data.
            if remainder and not re.search(r"[，,。；;：:]|(?:负责|要求|技能|训练|熟悉|具备|会用|懂)", remainder):
                fields["title"] = remainder.strip()
        if "后训练是训练一个招聘模型" in cleaned:
            facts.append(SemanticFact(
                value="后训练是训练一个招聘模型", category="responsibility", importance="neutral",
                source_type="explicit", evidence_text="后训练是训练一个招聘模型",
            ))
        if re.search(r"(?:技能(?:主要)?是)?懂\s*llm", cleaned, re.IGNORECASE):
            facts.append(SemanticFact(
                value="懂llm", category="requirement", importance="must",
                source_type="explicit", evidence_text="懂llm",
            ))
        if fields or facts:
            return StructuredCommand(
                intent="update", fields=JobFields.model_validate(fields), semantic_facts=facts,
            )
    return None


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
