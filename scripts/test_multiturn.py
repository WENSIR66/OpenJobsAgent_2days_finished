from __future__ import annotations

import asyncio

from backend.app.rag.service import CandidateRAGService


async def main() -> None:
    service = CandidateRAGService()
    conversation_id: str | None = None
    turns = [
        "寻找至少5年经验的Python后端工程师，有云平台经验优先。",
        "只看有AWS经验的候选人",
        "对比第1个和第2个，重点比较后端开发和云平台经验",
        "第一个为什么排在第一？",
    ]
    try:
        for index, message in enumerate(turns, start=1):
            response = await service.chat_turn(message, conversation_id)
            conversation_id = response.conversation_id
            print(f"\n===== TURN {index} =====")
            print("USER:", message)
            print("INTENT:", response.intent)
            print("CURRENT_QUERY:", response.current_query)
            print(
                "RESULT_IDS:",
                [candidate.candidate_id for candidate in response.candidates],
            )
            print("ANSWER:", response.answer[:1200])
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
