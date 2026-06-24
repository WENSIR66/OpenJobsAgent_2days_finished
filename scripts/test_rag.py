from __future__ import annotations

import argparse
import asyncio
import json

from backend.app.rag.service import CandidateRAGService


async def run(query: str) -> None:
    service = CandidateRAGService()
    try:
        result = await service.search(query)
    finally:
        await service.close()
    print("PARSED QUERY")
    print(result.parsed_query.model_dump_json(indent=2))
    print("\nSCORES")
    print(
        json.dumps(
            [
                {
                    "candidate_id": item.candidate_id,
                    "headline": item.metadata.get("headline"),
                    "vector": item.vector_score,
                    "bm25": item.bm25_score,
                    "should": item.metadata_should_score,
                    "final": item.final_score,
                }
                for item in result.candidates
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nANSWER")
    print(result.answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default="寻找至少5年经验的Python后端工程师，有云平台或DevOps经验优先",
    )
    args = parser.parse_args()
    asyncio.run(run(args.query))


if __name__ == "__main__":
    main()
