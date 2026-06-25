from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from backend.app.ingestion.config import Settings
from backend.app.rag.glm import AsyncGLMClient
from backend.app.rag.models import CandidateScore
from backend.app.rag.query_parser import QueryParser
from backend.app.rag.retriever import HybridRetriever

JUDGE_SYSTEM_PROMPT = """
你是候选人搜索排序评估 Judge。

任务：根据原始招聘 query，将给定候选人从最匹配到最不匹配排序。

规则：
1. 只依据原始 query 和候选人资料判断。
2. 优先满足 query 中的硬条件，再比较核心技能、相关经历、岗位方向和偏好条件。
3. 不得根据候选人所在公司名称推断资料中未明确写出的技能。
4. 候选人输入顺序已经随机打乱，不代表系统排序。
5. 你看不到系统原始排名和分数，不要猜测它们。
6. 每个候选人 ID 必须且只能出现一次。
7. 只输出严格 JSON，不要输出 Markdown、解释或其他字段：

{
  "ranked_candidate_ids": ["id1", "id2", "id3"]
}
"""

EVALUATION_NOTE = (
    "本评估使用 LLM-as-a-judge 对候选人 Top10 进行相对排序，"
    "Judge Top5 被视为弱监督强相关集合，用于快速评估系统排序质量。"
)


@dataclass(frozen=True)
class EvaluationPaths:
    queries: Path
    database: Path
    index_dir: Path
    json_report: Path
    markdown_report: Path


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def load_queries(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation query file must contain a JSON array")
    queries: list[str] = []
    for index, item in enumerate(payload, start=1):
        query = item.get("query") if isinstance(item, dict) else item
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Invalid query at position {index}")
        queries.append(query.strip())
    if not queries:
        raise ValueError("Evaluation query file is empty")
    return queries


def _truncate(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def candidate_for_judge(candidate: CandidateScore) -> dict[str, Any]:
    """Raw-profile projection only: no system rank, scores, or match evidence."""
    profile = candidate.profile
    experience = []
    for item in profile.get("experience", [])[:6]:
        experience.append(
            {
                "title": item.get("title"),
                "company_name": item.get("company_name"),
                "role": item.get("role"),
                "level": item.get("level"),
                "industry": item.get("industry"),
                "is_current": item.get("is_current"),
                "duration_months": item.get("duration_months"),
                "description": _truncate(item.get("description"), 700),
            }
        )
    education = [
        {
            "degree_str": item.get("degree_str"),
            "degree_level": item.get("degree_level"),
            "major": item.get("major"),
            "institution_name": item.get("institution_name"),
        }
        for item in profile.get("education", [])[:4]
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "headline": profile.get("headline"),
        "summary": _truncate(profile.get("summary"), 1000),
        "skills": profile.get("skills", []),
        "total_experience_duration_months": profile.get(
            "total_experience_duration_months"
        ),
        "experience": experience,
        "education": education,
        "certifications": profile.get("certifications", [])[:8],
    }


def validate_judge_ranking(
    response_text: str,
    expected_candidate_ids: Sequence[str],
) -> list[str]:
    payload = _extract_json(response_text)
    ranking = payload.get("ranked_candidate_ids")
    if not isinstance(ranking, list) or not all(
        isinstance(candidate_id, (str, int)) for candidate_id in ranking
    ):
        raise ValueError("ranked_candidate_ids must be an array of IDs")
    normalized = [str(candidate_id) for candidate_id in ranking]
    expected = [str(candidate_id) for candidate_id in expected_candidate_ids]
    if len(normalized) != len(expected):
        raise ValueError(
            f"Judge returned {len(normalized)} IDs; expected {len(expected)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Judge ranking contains duplicate candidate IDs")
    if set(normalized) != set(expected):
        missing = sorted(set(expected) - set(normalized))
        extra = sorted(set(normalized) - set(expected))
        raise ValueError(f"Judge ranking mismatch; missing={missing}, extra={extra}")
    return normalized


async def judge_candidates(
    client: AsyncGLMClient,
    query: str,
    shuffled_candidates: list[dict[str, Any]],
) -> tuple[list[str] | None, str | None]:
    candidate_ids = [str(item["candidate_id"]) for item in shuffled_candidates]
    user_content = json.dumps(
        {
            "query": query,
            "candidates": shuffled_candidates,
        },
        ensure_ascii=False,
    )
    last_error: str | None = None
    for _attempt in range(2):
        try:
            response = await client.complete(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return validate_judge_ranking(response, candidate_ids), None
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
    return None, last_error


def top5_overlap(system_ranking: Sequence[str], judge_ranking: Sequence[str]) -> float:
    system_top5 = set(system_ranking[:5])
    judge_top5 = set(judge_ranking[:5])
    return len(system_top5 & judge_top5) / 5.0


def binary_ndcg_at_10(
    system_ranking: Sequence[str],
    judge_relevant_set: set[str],
) -> float:
    gains = [
        1.0 if candidate_id in judge_relevant_set else 0.0
        for candidate_id in system_ranking[:10]
    ]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_relevant_count = min(len(judge_relevant_set), len(system_ranking), 10)
    if ideal_relevant_count == 0:
        return 0.0
    idcg = sum(
        1.0 / math.log2(index + 2) for index in range(ideal_relevant_count)
    )
    return dcg / idcg


def reciprocal_rank(
    system_ranking: Sequence[str],
    judge_best_candidate_id: str,
) -> float:
    try:
        return 1.0 / (system_ranking.index(judge_best_candidate_id) + 1)
    except ValueError:
        return 0.0


def compute_metrics(
    system_ranking: Sequence[str],
    judge_ranking: Sequence[str],
) -> dict[str, Any]:
    judge_top5 = list(judge_ranking[:5])
    judge_relevant_set = set(judge_top5)
    system_top5 = list(system_ranking[:5])
    return {
        "judge_top5": judge_top5,
        "top5_overlap": top5_overlap(system_ranking, judge_ranking),
        "ndcg_at_10": binary_ndcg_at_10(system_ranking, judge_relevant_set),
        "mrr": reciprocal_rank(system_ranking, judge_ranking[0]),
        "mismatch_cases": {
            "system_top5_but_not_judge_top5": [
                candidate_id
                for candidate_id in system_top5
                if candidate_id not in judge_relevant_set
            ],
            "judge_top5_but_not_system_top5": [
                candidate_id
                for candidate_id in judge_top5
                if candidate_id not in set(system_top5)
            ],
        },
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_results(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in query_results if not item["judge_failed"]]
    return {
        "total_queries": len(query_results),
        "evaluated_queries": len(successful),
        "judge_failed_queries": len(query_results) - len(successful),
        "mean_top5_overlap": _mean([item["top5_overlap"] for item in successful]),
        "mean_ndcg_at_10": _mean([item["ndcg_at_10"] for item in successful]),
        "mean_mrr": _mean([item["mrr"] for item in successful]),
    }


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def render_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Judge Ranking Evaluation",
        "",
        EVALUATION_NOTE,
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean_top5_overlap | {_format_metric(summary['mean_top5_overlap'])} |",
        f"| mean_ndcg_at_10 | {_format_metric(summary['mean_ndcg_at_10'])} |",
        f"| mean_mrr | {_format_metric(summary['mean_mrr'])} |",
        f"| evaluated_queries | {summary['evaluated_queries']} / {summary['total_queries']} |",
        f"| judge_failed_queries | {summary['judge_failed_queries']} |",
        "",
        "## Query Results",
        "",
    ]
    for index, result in enumerate(payload["query_results"], start=1):
        lines.extend(
            [
                f"### {index}. {result['query']}",
                "",
                f"- Judge failed: `{str(result['judge_failed']).lower()}`",
            ]
        )
        if result["judge_failed"]:
            lines.extend(
                [
                    f"- Judge error: `{result.get('judge_error') or 'unknown'}`",
                    f"- System Top10: `{result['system_top10']}`",
                    "",
                ]
            )
            continue
        mismatch = result["mismatch_cases"]
        lines.extend(
            [
                f"- Top5_Overlap: `{result['top5_overlap']:.4f}`",
                f"- nDCG@10: `{result['ndcg_at_10']:.4f}`",
                f"- MRR: `{result['mrr']:.4f}`",
                f"- System Top10: `{result['system_top10']}`",
                f"- LLM Judge Top10: `{result['judge_ranking']}`",
                f"- Judge Top5: `{result['judge_top5']}`",
                "- Mismatch cases:",
                "  - system_top5_but_not_judge_top5: "
                f"`{mismatch['system_top5_but_not_judge_top5']}`",
                "  - judge_top5_but_not_system_top5: "
                f"`{mismatch['judge_top5_but_not_system_top5']}`",
                "",
            ]
        )
    return "\n".join(lines)


async def evaluate(paths: EvaluationPaths, seed: int) -> dict[str, Any]:
    settings = Settings.from_env()
    client = AsyncGLMClient(settings)
    parser = QueryParser(client)
    retriever = HybridRetriever(paths.database, paths.index_dir, client)
    query_results: list[dict[str, Any]] = []
    queries = load_queries(paths.queries)
    try:
        for query_index, query in enumerate(queries):
            parsed = await parser.parse(query)
            candidates = await retriever.search(parsed, limit=10)
            system_top10 = [candidate.candidate_id for candidate in candidates]
            shuffled_candidates = [
                candidate_for_judge(candidate) for candidate in candidates
            ]
            random.Random(seed + query_index).shuffle(shuffled_candidates)

            if not shuffled_candidates:
                judge_ranking, judge_error = None, "System returned no candidates"
            else:
                judge_ranking, judge_error = await judge_candidates(
                    client,
                    query,
                    shuffled_candidates,
                )

            result: dict[str, Any] = {
                "query": query,
                "parsed_query": parsed.model_dump(),
                "system_top10": system_top10,
                "judge_ranking": judge_ranking,
                "judge_top5": [],
                "top5_overlap": None,
                "ndcg_at_10": None,
                "mrr": None,
                "mismatch_cases": {
                    "system_top5_but_not_judge_top5": [],
                    "judge_top5_but_not_system_top5": [],
                },
                "judge_failed": judge_ranking is None,
                "judge_error": judge_error,
            }
            if judge_ranking is not None:
                result.update(compute_metrics(system_top10, judge_ranking))
            query_results.append(result)
            status = "FAILED" if result["judge_failed"] else "OK"
            print(f"[{query_index + 1}/{len(queries)}] {status}: {query}")
    finally:
        await client.close()

    payload = {
        "evaluation": "llm_judge_ranking",
        "generated_at": datetime.now(UTC).isoformat(),
        "judge_model": settings.chat_model,
        "retrieval_embedding_model": settings.embedding_model,
        "random_seed": seed,
        "methodology_note": EVALUATION_NOTE,
        "summary": summarize_results(query_results),
        "query_results": query_results,
    }
    paths.json_report.parent.mkdir(parents=True, exist_ok=True)
    paths.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    paths.json_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.markdown_report.write_text(
        render_markdown_report(payload),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate ranking with a blinded LLM judge"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval_queries.json"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/candidates.db"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/indexes"),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/judge_ranking_eval_results.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/judge_ranking_eval_report.md"),
    )
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = EvaluationPaths(
        queries=args.queries,
        database=args.database,
        index_dir=args.index_dir,
        json_report=args.json_report,
        markdown_report=args.markdown_report,
    )
    payload = asyncio.run(evaluate(paths, args.seed))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON report: {paths.json_report}")
    print(f"Markdown report: {paths.markdown_report}")


if __name__ == "__main__":
    main()
