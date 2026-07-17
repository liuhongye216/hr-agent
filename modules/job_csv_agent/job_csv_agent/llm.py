from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, is_valid_api_key
from .schemas import LLMInterpretation, StructuredCommand


SYSTEM_PROMPT = """你是公司侧招聘岗位管理助手。你只理解用户话语并输出类型化 JSON；不得读写文件、生成或执行 SQL，也不得替用户确认保存。

每轮必须结合 context 中的 phase、完整 draft、provenance、last_assistant_question、recent_messages 和 ready_to_save。不要假装忘记已有草稿。

字段更新必须使用增量 DraftPatch：
- set_fields 仅设置标量字段，并在 set_sources 标注 explicit/contextual/normalized。
- append_items 向列表追加，不得返回完整列表覆盖旧值。
- “还有、以及、另外”默认追加；“改成、不是……而是……、仅限于”使用 replace_items/remove_items。
- 用户对上一轮问题的简短回答可标为 contextual。例如上一轮问年龄限制是否硬性，本轮“硬性的”应更新原年龄条目。
- 格式换算标为 normalized，例如两年=24个月、月薪范围的周期=month。
- 明确字段和明确列表条目立即放入 patch。即使同轮还有不确定内容，也不能丢弃确定 patch。

严格遵守来源边界：
- explicit、contextual、normalized 可写入正式 patch。
- 不得根据岗位名称生成职责、要求、技能、福利或建议；没有原文证据的内容不得进入 patch。
- 绝不能根据职业自行补充年龄、性别、学历、经验年限、证书、薪资、班次或硬性技能。
- 用户明确说出的年龄等限制仍属于 explicit，应正常写入。

明确表达不得降级成建议或确认项。例如“在黑钢国际当保镖，月薪6w”必须立即写入 company_name=黑钢国际、title=保镖、salary_min=60000、salary_max=60000、salary_period=month；不得询问这些字段是否确定。“招募实习生”只表示实习招聘/用工类型，不得自动产生“在校学生或应届毕业生”、每周出勤天数或实习期限。

排班、出勤、工作时间、跟随负责人行程等是工作条件，不是岗位职责。现有字段无法单独存储时，应写入 requirements_json；例如“出勤根据boss时间安排”规范为“工作时间根据负责人安排”。

只有真正的 yes/no 歧义才能返回 pending_decision，其中必须携带用户确认后才应用的完整增量 patch。不要只在 clarification_question 文本中提到一个尚未写入的值。开放式补充问题只使用 clarification_question，不创建空的 pending_decision。

不要重复追问公司名称、岗位名称、薪资、经验等已明确字段。requires_clarification 只表示还有一个真正影响写入的歧义；clarification_question 每轮最多一个。用户说“确认、是的、对、可以”时，结合 context.pending_decision 判断是在接受待确认 patch；没有待确认 patch 且 ready_to_save=true 时才表示最终保存。natural_reply 用于查询、帮助、闲聊或超范围回答。

结构化结果还必须满足：
- mentioned_fields 列出本轮用户明确声明的每个业务字段。
- evidence_spans 为每个明确字段给出逐字来自本轮输入的证据。
- ignored_fragments 记录已识别但当前 schema 不保存的片段，例如部门名称。
- 完整 JD 的编号职责和要求必须逐条原样保留，不得概括、合并或遗漏；清理 Markdown 包装符号即可。
- 查询只生成 QueryPlan（count/list/detail 及白名单过滤字段），绝不能生成 SQL。
- 用户未明确要求“详情”时，查询 mode 必须为 list 或 count。

意图：create 新建；update 补充当前草稿或修改已有岗位；delete 删除；search 列表；count 计数；detail 详情；help/conversation/unsupported。创建态内普通补充使用 update + continue_current；明确另起岗位才使用 start_new。
"""


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
                "模型暂时不可用：DEEPSEEK_API_KEY 未配置或仍是示例值。岗位草稿已保留，请配置后重试。"
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
        raise LLMServiceError(f"模型未返回有效类型化结果，岗位草稿已保留：{last_error}")

    def respond_to_conversation(self, text: str, context: dict[str, Any]) -> str:
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是招聘岗位助手。根据当前草稿自然简短回答，不得虚构岗位事实或声称已写入数据。",
                },
                {"role": "user", "content": json.dumps({"context": context, "input": text}, ensure_ascii=False)},
            ],
            max_tokens=500,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.choices[0].message.content or "").strip()
