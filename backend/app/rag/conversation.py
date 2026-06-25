from __future__ import annotations

import asyncio
import json
import re
from uuid import uuid4

from .glm import AsyncGLMClient
from .models import CandidateScore, ConversationState, IntentDecision

INTENT_SYSTEM_PROMPT = """
你是候选人搜索 Agent 的多轮意图识别器。只输出 JSON，不要解释。

输出：
{
  "intent": "new_search|refine|compare|follow_up",
  "rewritten_query": "必要时给出结合上下文后的完整查询，否则为 null",
  "candidate_refs": ["1", "3", "候选人ID"]
}

意图定义：
- new_search：明显切换招聘目标，或要求重新寻找另一类岗位。
- refine：只看、再筛、在刚才结果里找、增加技能/学历/年限/行业条件，或补充新的语义要求。
- compare：要求比较两个或多个当前候选人。
- follow_up：询问某个候选人为何排名靠前、哪里符合，或询问当前结果整体情况。

规则：
1. refine 的 rewritten_query 必须把 current_query 和用户新增要求合并成完整独立查询。
2. new_search 的 rewritten_query 就是当前用户的新目标。
3. compare/follow_up 通常不重写查询。
4. “第1个和第3个”输出 ["1", "3"]；候选人 ID 原样输出。
5. “这两个谁更合适”没有明确编号时，可输出 ["1", "2"]。
"""


class ConversationStore:
    """In-memory state; process restart naturally clears all conversations."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, conversation_id: str | None
    ) -> tuple[str, ConversationState]:
        async with self._lock:
            resolved_id = conversation_id or str(uuid4())
            state = self._states.setdefault(resolved_id, ConversationState())
            return resolved_id, state

    async def update(
        self,
        conversation_id: str,
        *,
        current_query: str | None,
        current_results: list[CandidateScore],
    ) -> ConversationState:
        state = ConversationState(
            current_query=current_query,
            current_results=current_results,
        )
        async with self._lock:
            self._states[conversation_id] = state
        return state

    async def clear(self, conversation_id: str) -> None:
        async with self._lock:
            self._states[conversation_id] = ConversationState()


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_candidate_refs(message: str) -> list[str]:
    refs: list[str] = []
    for match in re.findall(r"第\s*(\d+)\s*(?:个|位|名)?", message):
        if match not in refs:
            refs.append(match)
    chinese_positions = {
        "一": "1",
        "二": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "五": "5",
    }
    for match in re.findall(r"第\s*([一二两三四五])\s*(?:个|位|名)?", message):
        resolved = chinese_positions[match]
        if resolved not in refs:
            refs.append(resolved)
    for match in re.findall(r"(?<!\d)(\d{7,})(?!\d)", message):
        if match not in refs:
            refs.append(match)
    if not refs and re.search(r"这两个|两个人|二者", message):
        return ["1", "2"]
    return refs


def _fallback_intent(
    message: str,
    state: ConversationState,
) -> IntentDecision:
    if not state.current_query or not state.current_results:
        return IntentDecision(
            intent="new_search",
            rewritten_query=message,
        )
    refs = _extract_candidate_refs(message)
    if re.search(r"对比|比较|谁更合适|哪个好|优劣", message):
        return IntentDecision(intent="compare", candidate_refs=refs)
    if re.search(r"重新找|重新搜索|换成|改找|另找|新查询", message):
        return IntentDecision(
            intent="new_search",
            rewritten_query=message,
        )
    if re.search(
        r"^(?:帮我)?(?:找|寻找|搜索).*(?:工程师|产品经理|分析师|设计师|架构师|开发者)",
        message,
    ) and not re.search(r"在刚才|从这些|只看", message):
        return IntentDecision(
            intent="new_search",
            rewritten_query=message,
        )
    if re.search(
        r"只看|再筛|筛一下|在刚才|从这些|限定|排除|还要|同时要|再加上|增加条件",
        message,
    ):
        return IntentDecision(
            intent="refine",
            rewritten_query=f"{state.current_query}；新增要求：{message}",
            candidate_refs=refs,
        )
    return IntentDecision(intent="follow_up", candidate_refs=refs)


class IntentRouter:
    def __init__(self, client: AsyncGLMClient) -> None:
        self.client = client

    async def classify(
        self,
        message: str,
        state: ConversationState,
    ) -> IntentDecision:
        if not state.current_query or not state.current_results:
            return IntentDecision(
                intent="new_search",
                rewritten_query=message,
            )
        result_summary = [
            {
                "position": index,
                "candidate_id": candidate.candidate_id,
                "headline": candidate.metadata.get("headline"),
            }
            for index, candidate in enumerate(state.current_results, start=1)
        ]
        try:
            content = await self.client.complete(
                [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"current_query：{state.current_query}\n"
                            f"current_results：{json.dumps(result_summary, ensure_ascii=False)}\n"
                            f"用户新输入：{message}"
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            decision = IntentDecision.model_validate(_extract_json(content))
            if decision.intent == "refine" and not decision.rewritten_query:
                decision.rewritten_query = (
                    f"{state.current_query}；新增要求：{message}"
                )
            if decision.intent == "new_search" and not decision.rewritten_query:
                decision.rewritten_query = message
            if decision.intent in {"compare", "follow_up"} and not decision.candidate_refs:
                decision.candidate_refs = _extract_candidate_refs(message)
            return decision
        except Exception:
            return _fallback_intent(message, state)


def resolve_candidate_refs(
    refs: list[str],
    current_results: list[CandidateScore],
) -> list[CandidateScore]:
    resolved: list[CandidateScore] = []
    by_id = {candidate.candidate_id: candidate for candidate in current_results}
    for ref in refs:
        candidate: CandidateScore | None = None
        if ref in by_id:
            candidate = by_id[ref]
        elif ref.isdigit():
            position = int(ref)
            if 1 <= position <= len(current_results):
                candidate = current_results[position - 1]
        if candidate and candidate.candidate_id not in {
            item.candidate_id for item in resolved
        }:
            resolved.append(candidate)
    return resolved
