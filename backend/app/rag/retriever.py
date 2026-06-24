from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.app.ingestion.storage import make_fts_query

from .glm import AsyncGLMClient
from .models import CandidateScore, MetadataCondition, ParsedQuery

SCALAR_FIELDS = {
    "is_working",
    "current_title",
    "current_department",
    "management_level",
    "is_decision_maker",
    "total_experience_months",
    "total_experience_years",
    "highest_degree_level",
}
LIST_COLUMNS = {
    "skills": "skills_json",
    "roles": "roles_json",
    "levels": "levels_json",
    "industries": "industries_json",
    "companies": "companies_json",
    "locations": "locations_json",
    "majors": "majors_json",
    "education_countries": "metadata_json",
}


def _keyword_sql(values: list[Any]) -> tuple[str, list[Any]]:
    alias_clauses: list[str] = []
    params: list[Any] = []
    for value in values:
        alias = str(value).casefold().strip()
        if not alias:
            continue
        alias_clauses.append("LOWER(page_content) LIKE ?")
        params.append(f"%{alias}%")
    if not alias_clauses:
        raise ValueError("Keyword condition contains no searchable terms")
    return "(" + " OR ".join(alias_clauses) + ")", params


def _sql_condition(condition: MetadataCondition) -> tuple[str, list[Any]]:
    field, operator, value = condition.field, condition.operator, condition.value
    if field in {"skills", "roles"} and operator in {"contains", "in"}:
        values = value if operator == "in" and isinstance(value, list) else [value]
        return _keyword_sql(values)
    if field in SCALAR_FIELDS:
        if operator == "eq":
            return f"{field} = ?", [int(value) if isinstance(value, bool) else value]
        if operator in {"gte", "lte"}:
            return f"{field} {'>=' if operator == 'gte' else '<='} ?", [value]
        if operator == "contains":
            return f"LOWER(COALESCE({field}, '')) LIKE ?", [f"%{str(value).lower()}%"]
        if operator == "in" and isinstance(value, list) and value:
            placeholders = ",".join("?" for _ in value)
            return f"{field} IN ({placeholders})", value
    if field in LIST_COLUMNS:
        if field == "education_countries":
            expression = (
                "EXISTS (SELECT 1 FROM json_each(metadata_json, '$.education_countries') "
                "WHERE LOWER(CAST(value AS TEXT)) LIKE ?)"
            )
            return expression, [f"%{str(value).lower()}%"]
        column = LIST_COLUMNS[field]
        values = value if operator == "in" and isinstance(value, list) else [value]
        clauses = []
        params: list[Any] = []
        for item in values:
            clauses.append(
                f"EXISTS (SELECT 1 FROM json_each({column}) "
                "WHERE LOWER(CAST(value AS TEXT)) LIKE ?)"
            )
            params.append(f"%{str(item).lower()}%")
        return "(" + " OR ".join(clauses) + ")", params
    raise ValueError(f"Unsupported metadata condition: {condition}")


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _condition_matches(metadata: dict[str, Any], condition: MetadataCondition) -> bool:
    actual = metadata.get(condition.field)
    expected = condition.value
    if actual is None:
        return False
    if condition.operator == "eq":
        return actual == expected or str(actual).casefold() == str(expected).casefold()
    if condition.operator == "gte":
        return float(actual) >= float(expected)
    if condition.operator == "lte":
        return float(actual) <= float(expected)
    values = actual if isinstance(actual, list) else [actual]
    expected_values = (
        expected
        if condition.operator == "in" and isinstance(expected, list)
        else [expected]
    )
    return any(
        str(needle).casefold() in str(value).casefold()
        for value in values
        for needle in expected_values
    )


def _text_match(page_content: str, condition: MetadataCondition) -> str | None:
    values = condition.value if isinstance(condition.value, list) else [condition.value]
    text = page_content.casefold()
    for value in values:
        alias = str(value).casefold().strip()
        if not alias:
            continue
        if re.fullmatch(r"[a-z0-9]+", alias):
            matched = re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                text,
            )
        else:
            matched = alias in text
        if matched:
            return str(value)
    return None


def _candidate_condition_match(
    metadata: dict[str, Any],
    page_content: str,
    condition: MetadataCondition,
) -> tuple[bool, str]:
    if condition.field in {"skills", "roles"}:
        matched = _text_match(page_content, condition)
        if matched:
            return True, f"候选人完整履历文本命中“{matched}”"
        return False, f"候选人完整履历文本未命中 {condition.value}"
    matched = _condition_matches(metadata, condition)
    return matched, (
        f"metadata 满足 {condition.field} {condition.operator} {condition.value}"
        if matched
        else f"metadata 不满足 {condition.field} {condition.operator} {condition.value}"
    )


class HybridRetriever:
    def __init__(
        self,
        database_path: Path,
        index_dir: Path,
        glm_client: AsyncGLMClient,
    ) -> None:
        self.database_path = database_path
        self.embedding_client = glm_client
        self.index = faiss.read_index(str(index_dir / "candidates.faiss"))
        manifest = json.loads(
            (index_dir / "candidates.manifest.json").read_text(encoding="utf-8")
        )
        self.candidate_ids: list[str] = manifest["candidate_ids"]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _allowed_ids(self, conditions: list[MetadataCondition]) -> set[str]:
        clauses, params = [], []
        for condition in conditions:
            clause, values = _sql_condition(condition)
            clauses.append(clause)
            params.extend(values)
        sql = "SELECT candidate_id, page_content FROM candidates"
        if clauses:
            sql += " WHERE " + " AND ".join(f"({clause})" for clause in clauses)
        with self._connect() as connection:
            rows = connection.execute(sql, params)
            keyword_conditions = [
                condition
                for condition in conditions
                if condition.field in {"skills", "roles"}
                and condition.operator in {"contains", "in"}
            ]
            return {
                row["candidate_id"]
                for row in rows
                if all(
                    _text_match(row["page_content"], condition)
                    for condition in keyword_conditions
                )
            }

    async def _vector_search(
        self, query: str, allowed: set[str], limit: int
    ) -> dict[str, float]:
        vector = np.asarray(await self.embedding_client.embed([query]), dtype="float32")
        faiss.normalize_L2(vector)
        scores, indexes = await asyncio.to_thread(
            self.index.search, vector, self.index.ntotal
        )
        output: dict[str, float] = {}
        for score, index in zip(scores[0], indexes[0], strict=True):
            if index < 0:
                continue
            candidate_id = self.candidate_ids[index]
            if candidate_id in allowed:
                output[candidate_id] = float(score)
                if len(output) == limit:
                    break
        return output

    def _bm25_search(self, query: str, allowed: set[str], limit: int) -> dict[str, float]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id, -bm25(candidate_fts) AS score
                FROM candidate_fts
                WHERE candidate_fts MATCH ?
                ORDER BY bm25(candidate_fts)
                """,
                (make_fts_query(query),),
            )
            output: dict[str, float] = {}
            for row in rows:
                if row["candidate_id"] in allowed:
                    output[row["candidate_id"]] = float(row["score"])
                    if len(output) == limit:
                        break
        return output

    def _load_candidates(self, candidate_ids: set[str]) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in candidate_ids)
        with self._connect() as connection:
            return connection.execute(
                f"""
                SELECT candidate_id, metadata_json, clean_json, page_content
                FROM candidates WHERE candidate_id IN ({placeholders})
                """,
                list(candidate_ids),
            ).fetchall()

    async def search(self, parsed: ParsedQuery, limit: int = 5) -> list[CandidateScore]:
        allowed = await asyncio.to_thread(
            self._allowed_ids, parsed.metadata_filter_must
        )
        if not allowed:
            return []

        raw_vector, raw_bm25 = await asyncio.gather(
            self._vector_search(parsed.semantic_query, allowed, 20),
            asyncio.to_thread(
                self._bm25_search, parsed.semantic_query, allowed, 20
            ),
        )
        vector = _normalize(raw_vector)
        bm25 = _normalize(raw_bm25)
        candidate_ids = set(vector) | set(bm25)
        if not candidate_ids:
            return []

        rows = await asyncio.to_thread(self._load_candidates, candidate_ids)
        results: list[CandidateScore] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            profile = json.loads(row["clean_json"])
            page_content = row["page_content"]
            matched_must: dict[str, bool] = {}
            for condition in parsed.metadata_filter_must:
                matched, _ = _candidate_condition_match(
                    metadata, page_content, condition
                )
                matched_must[
                    f"{condition.field} {condition.operator} {condition.value}"
                ] = matched

            matched_should: list[str] = []
            for condition in parsed.metadata_filter_should:
                matched, detail = _candidate_condition_match(
                    metadata, page_content, condition
                )
                if matched:
                    matched_should.append(detail)
            should_score = (
                len(matched_should) / len(parsed.metadata_filter_should)
                if parsed.metadata_filter_should
                else 0.0
            )
            candidate_id = row["candidate_id"]
            final = (
                0.35 * vector.get(candidate_id, 0.0)
                + 0.35 * bm25.get(candidate_id, 0.0)
                + 0.30 * should_score
            )
            results.append(
                CandidateScore(
                    candidate_id=candidate_id,
                    vector_score=round(vector.get(candidate_id, 0.0), 4),
                    bm25_score=round(bm25.get(candidate_id, 0.0), 4),
                    metadata_should_score=round(should_score, 4),
                    final_score=round(final, 4),
                    matched_must=matched_must,
                    matched_should=matched_should,
                    metadata=metadata,
                    profile=profile,
                )
            )
        return sorted(results, key=lambda item: item.final_score, reverse=True)[:limit]
