from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, is_valid_api_key
from .schemas import (
    ConstrainedRewriteBatch, LLMInterpretation, Phase, SemanticFact, StructuredCommand,
)


SYSTEM_PROMPT = """你是公司侧招聘岗位管理助手。你的职责只是理解用户话语并输出类型化 JSON；
绝不能读写文件、生成或执行 SQL、调用 Repository，也不能替用户确认任何写操作。

每轮必须先结合 context.phase、当前草稿、待处理动作和候选数量判断路由与动作，再抽取字段。
路由包括 job_write、job_read、task_control、help、conversation、unsupported；动作必须与 intent 一致。
在 CREATING 中，区分五类输入：补充当前岗位、闲聊、帮助、查询已有岗位、明确发起的新任务。
闲聊和帮助不得改变草稿；查询只读；新任务不得被当成当前岗位字段。

不得用关键词删除、前后缀截取或正则猜岗位名。必须理解完整语义：
- “我要招聘”只表示开始创建岗位，title 必须为 null；不得把“聘”当成岗位名。
- “我要招聘算法工程师”表示创建，title 必须是完整的“算法工程师”。
- 用户没有明确提供的字段保持 null，不得凭常识补齐。

输出必须包含 route（category/action/confidence/reason）、intent、task_relation、fields、field_evidence、
semantic_facts、confidence、requires_clarification、clarification_questions、coverage 和 natural_reply。
在创建态补充当前草稿用 continue_current；用户明确另起岗位任务用 start_new；其余用 not_applicable。
每个非列表字段都要有 field_evidence，证据必须逐字来自本轮输入；推断值不能进入 fields。
低置信度或存在关键歧义时 requires_clarification=true，fields 只保留无歧义内容，并给出自然追问。

JD 处理完全由你完成：识别章节和条目，拆分原子事实，区分 responsibility、requirement、skill、
benefit、unknown，并在不增强条件的前提下改写成自然、可发布的文本。每条 semantic_fact 必须引用
本轮原文 evidence_text；保留 source_section 和 source_item_index。explicit 且分类明确的事实才可投影；
inferred/unknown 必须 needs_confirmation=true 且不得进入 fields。职责是入职后执行的工作；要求是候选人
入职前应具备的条件；技能字段只放原文明确出现的简短标签；福利是公司提供的待遇。
不得新增原文没有的学历、年限、证书、技术、薪资、熟练度或“必须/精通/独立负责”等强度。
coverage 应覆盖识别到的每个 JD 条目，并保留原文、章节、顺序和字符位置；无法可靠判断时标为
needs_clarification。natural_reply 用于帮助、闲聊、澄清或超范围回复，应自然简短且不声称已写入。
"""


def deterministic_command(
    text: str, phase: str | Phase = Phase.IDLE.value,
) -> StructuredCommand | None:
    """Deprecated compatibility hook; business-language rules are disabled."""
    return None


class LLMConfigurationError(RuntimeError):
    pass


class LLMServiceError(RuntimeError):
    pass


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        return content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return content


class StructuredInterpreter:
    def __init__(self, model: str = DEEPSEEK_MODEL) -> None:
        self.model = model

    @staticmethod
    def _client() -> Any:
        if not is_valid_api_key(DEEPSEEK_API_KEY):
            raise LLMConfigurationError(
                "模型暂时不可用：DEEPSEEK_API_KEY 未配置或仍是示例值。岗位草稿已保留，"
                "请配置模型后重试。"
            )
        from openai import OpenAI

        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def extract(self, text: str, context: dict[str, Any]) -> StructuredCommand:
        from openai import (
            APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, RateLimitError,
        )

        schema = json.dumps(LLMInterpretation.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n只输出符合此 schema 的 JSON：{schema}"},
            {"role": "user", "content": json.dumps({"context": context, "input": text}, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = self._client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=4_000,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except AuthenticationError as exc:
                raise LLMConfigurationError("模型鉴权失败，岗位草稿已保留，请检查 API 密钥。") from exc
            except BadRequestError as exc:
                raise LLMServiceError(f"模型拒绝请求，请检查模型名 {self.model}。岗位草稿已保留。") from exc
            except RateLimitError as exc:
                raise LLMServiceError("模型当前限流或额度不足，岗位草稿已保留，请稍后重试。") from exc
            except APIConnectionError as exc:
                raise LLMServiceError("暂时无法连接模型，岗位草稿已保留，请稍后重试。") from exc
            except APIStatusError as exc:
                raise LLMServiceError(f"模型服务返回异常状态 {exc.status_code}，岗位草稿已保留。") from exc
            content = _strip_json_fence(response.choices[0].message.content or "")
            try:
                return LLMInterpretation.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                messages.extend([
                    {"role": "assistant", "content": content or "{}"},
                    {"role": "user", "content": f"JSON 不符合类型化协议：{exc}。请修正后完整输出。"},
                ])
        raise LLMServiceError(f"模型未返回有效的类型化结果，岗位草稿已保留：{last_error}")

    def rewrite_semantic_facts(
        self, facts: list[SemanticFact], original_text: str,
    ) -> ConstrainedRewriteBatch:
        """Optional LLM copy editor; the agent does not need it for interpretation."""
        if not facts:
            return ConstrainedRewriteBatch()
        schema = json.dumps(ConstrainedRewriteBatch.model_json_schema(), ensure_ascii=False)
        payload = [
            {
                "index": index,
                "value": fact.value,
                "category": fact.category,
                "importance": fact.importance,
                "evidence": fact.evidence_texts or [fact.evidence_text],
            }
            for index, fact in enumerate(facts)
            if fact.source_type in {"explicit", "user_confirmed"} and fact.category != "unknown"
        ]
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 JD 受约束改写器。只能基于给定 evidence 改写，不得新增技术、学历、年限、"
                        "证书、薪资或能力强度；必须用 evidence_indices 逐条引用。只输出 schema JSON：" + schema
                    ),
                },
                {"role": "user", "content": json.dumps({"source": original_text, "facts": payload}, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            max_tokens=4_000,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return ConstrainedRewriteBatch.model_validate_json(
            _strip_json_fence(response.choices[0].message.content or "{}")
        )

    def respond_to_conversation(self, text: str, context: dict[str, Any]) -> str:
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是招聘岗位助手。根据任务状态自然简短回复；不得虚构岗位事实，"
                        "不得声称已读写数据或替用户确认。"
                    ),
                },
                {"role": "user", "content": json.dumps({"context": context, "input": text}, ensure_ascii=False)},
            ],
            max_tokens=500,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.choices[0].message.content or "").strip()
