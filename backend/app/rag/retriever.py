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

FIELD_ROUTE_SOURCES = {
    "skills": (
        "skills",
        "experience_descriptions",
        "summary",
        "experience_titles",
        "headline",
    ),
    "roles": (
        "current_title",
        "experience_titles",
        "headline",
        "roles",
    ),
    "industries": (
        "industries",
        "company_tags",
        "experience_descriptions",
        "summary",
    ),
    "companies": (
        "companies",
        "experience_descriptions",
        "summary",
    ),
    "locations": (
        "locations",
        "experience_addresses",
        "summary",
    ),
    "majors": (
        "majors",
        "degree_descriptions",
        "courses",
    ),
    "certifications": (
        "certifications",
        "summary",
        "experience_descriptions",
    ),
    "experience_descriptions": ("experience_descriptions",),
}


def _condition_values(condition: MetadataCondition) -> list[str]:
    values = (
        condition.value
        if condition.operator == "in" and isinstance(condition.value, list)
        else [condition.value]
    )
    return [str(value).casefold().strip() for value in values if str(value).strip()]


def _json_array_sql(column: str, values: list[str]) -> tuple[str, list[Any]]:
    clauses = [
        f"EXISTS (SELECT 1 FROM json_each({column}) "
        "WHERE LOWER(CAST(value AS TEXT)) LIKE ?)"
        for _ in values
    ]
    return "(" + " OR ".join(clauses) + ")", [f"%{value}%" for value in values]


def _json_experience_sql(json_path: str, values: list[str]) -> tuple[str, list[Any]]:
    clauses = [
        "EXISTS (SELECT 1 FROM json_each(clean_json, '$.experience') "
        f"WHERE LOWER(COALESCE(json_extract(value, '{json_path}'), '')) LIKE ?)"
        for _ in values
    ]
    return "(" + " OR ".join(clauses) + ")", [f"%{value}%" for value in values]


def _source_sql(source: str, values: list[str]) -> tuple[str, list[Any]]:
    if source == "skills":
        return _json_array_sql("skills_json", values)
    if source == "roles":
        return _json_array_sql("roles_json", values)
    if source == "industries":
        return _json_array_sql("industries_json", values)
    if source == "companies":
        return _json_array_sql("companies_json", values)
    if source == "locations":
        return _json_array_sql("locations_json", values)
    if source == "majors":
        return _json_array_sql("majors_json", values)
    if source == "experience_titles":
        return _json_experience_sql("$.title", values)
    if source == "experience_descriptions":
        return _json_experience_sql("$.description", values)
    if source == "company_tags":
        return _json_experience_sql("$.company_tags", values)
    if source == "experience_addresses":
        clauses: list[str] = []
        params: list[Any] = []
        for path in (
            "$.full_address",
            "$.address_city",
            "$.address_state",
            "$.address_country",
        ):
            sql, sql_params = _json_experience_sql(path, values)
            clauses.append(sql)
            params.extend(sql_params)
        return "(" + " OR ".join(clauses) + ")", params
    if source in {"summary", "headline"}:
        clauses = [
            f"LOWER(COALESCE(json_extract(clean_json, '$.{source}'), '')) LIKE ?"
            for _ in values
        ]
        return "(" + " OR ".join(clauses) + ")", [f"%{value}%" for value in values]
    if source == "current_title":
        clauses = [
            "LOWER(COALESCE(current_title, '')) LIKE ?" for _ in values
        ]
        return "(" + " OR ".join(clauses) + ")", [f"%{value}%" for value in values]
    if source == "degree_descriptions":
        clauses = [
            "EXISTS (SELECT 1 FROM json_each(clean_json, '$.education') "
            "WHERE LOWER(COALESCE(json_extract(value, '$.degree_str'), '')) LIKE ?)"
            for _ in values
        ]
        return "(" + " OR ".join(clauses) + ")", [f"%{value}%" for value in values]
    if source == "courses":
        return _json_array_sql("json_extract(clean_json, '$.courses')", values)
    if source == "certifications":
        return _json_array_sql("json_extract(clean_json, '$.certifications')", values)
    raise ValueError(f"Unsupported field route source: {source}")


def _routed_sql(field: str, values: list[str]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for source in FIELD_ROUTE_SOURCES[field]:
        clause, source_params = _source_sql(source, values)
        clauses.append(clause)
        params.extend(source_params)
    return "(" + " OR ".join(clauses) + ")", params


def _sql_condition(condition: MetadataCondition) -> tuple[str, list[Any]]:
    field, operator, value = condition.field, condition.operator, condition.value
    if operator in {"contains", "in"}:
        values = _condition_values(condition)
        if not values:
            raise ValueError("Keyword condition contains no searchable terms")
        if field in FIELD_ROUTE_SOURCES:
            return _routed_sql(field, values)
        if field == "experience_titles":
            return _json_experience_sql("$.title", values)
        if field == "summary":
            clauses = [
                "LOWER(COALESCE(json_extract(clean_json, '$.summary'), '')) LIKE ?"
                for _ in values
            ]
            return "(" + " OR ".join(clauses) + ")", [f"%{item}%" for item in values]
        if field == "headline":
            clauses = [
                "LOWER(COALESCE(headline, '')) LIKE ?" for _ in values
            ]
            return "(" + " OR ".join(clauses) + ")", [f"%{item}%" for item in values]
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


def _text_match(text_value: str, condition: MetadataCondition) -> str | None:
    values = _condition_values(condition)
    text = text_value.casefold()
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


def _stringify_items(values: list[Any]) -> list[str]:
    return [
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list))
        else str(value)
        for value in values
        if value not in (None, "", [], {})
    ]


def _profile_source_values(
    metadata: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, list[str]]:
    experience = profile.get("experience", [])
    education = profile.get("education", [])
    return {
        "skills": _stringify_items(profile.get("skills", [])),
        "roles": _stringify_items(metadata.get("roles", [])),
        "industries": _stringify_items(metadata.get("industries", [])),
        "companies": _stringify_items(metadata.get("companies", [])),
        "locations": _stringify_items(metadata.get("locations", [])),
        "majors": _stringify_items(metadata.get("majors", [])),
        "current_title": _stringify_items(
            [
                profile.get("active_experience_title"),
                next(
                    (
                        item.get("title")
                        for item in experience
                        if item.get("is_current") and item.get("title")
                    ),
                    None,
                ),
            ]
        ),
        "experience_titles": _stringify_items(
            [item.get("title") for item in experience]
        ),
        "experience_descriptions": _stringify_items(
            [item.get("description") for item in experience]
        ),
        "company_tags": _stringify_items(
            [
                tag
                for item in experience
                for tag in (item.get("company_tags") or [])
            ]
        ),
        "experience_addresses": _stringify_items(
            [
                value
                for item in experience
                for value in (
                    item.get("full_address"),
                    item.get("address_city"),
                    item.get("address_state"),
                    item.get("address_country"),
                )
            ]
        ),
        "degree_descriptions": _stringify_items(
            [item.get("degree_str") for item in education]
        ),
        "courses": _stringify_items(profile.get("courses", [])),
        "certifications": _stringify_items(profile.get("certifications", [])),
        "summary": _stringify_items([profile.get("summary")]),
        "headline": _stringify_items([profile.get("headline")]),
    }


def _candidate_condition_match(
    metadata: dict[str, Any],
    profile: dict[str, Any],
    condition: MetadataCondition,
) -> tuple[bool, str]:
    source_values = _profile_source_values(metadata, profile)
    if condition.field in FIELD_ROUTE_SOURCES:
        routed_values = [
            (source, text)
            for source in FIELD_ROUTE_SOURCES[condition.field]
            for text in source_values[source]
        ]
        matched = next(
            (
                (source, alias)
                for source, text in routed_values
                if (alias := _text_match(text, condition))
            ),
            None,
        )
        if matched:
            source, alias = matched
            return True, f"{source} 字段明确命中“{alias}”"
        return False, f"{condition.field} 字段未命中 {condition.value}"
    if condition.field in {"experience_titles", "headline", "summary"}:
        matched = next(
            (
                alias
                for text in source_values[condition.field]
                if (alias := _text_match(text, condition))
            ),
            None,
        )
        if matched:
            return True, f"{condition.field} 字段明确命中“{matched}”"
        return False, f"{condition.field} 字段未命中 {condition.value}"
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
        sql = "SELECT candidate_id, metadata_json, clean_json FROM candidates"
        if clauses:
            sql += " WHERE " + " AND ".join(f"({clause})" for clause in clauses)
        with self._connect() as connection:
            rows = connection.execute(sql, params)
            return {
                row["candidate_id"]
                for row in rows
                if all(
                    _candidate_condition_match(
                        json.loads(row["metadata_json"]),
                        json.loads(row["clean_json"]),
                        condition,
                    )[0]
                    for condition in conditions
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
                SELECT candidate_id, metadata_json, clean_json
                FROM candidates WHERE candidate_id IN ({placeholders})
                """,
                list(candidate_ids),
            ).fetchall()

    async def search(
        self,
        parsed: ParsedQuery,
        limit: int = 5,
        candidate_scope: set[str] | None = None,
    ) -> list[CandidateScore]:
        allowed = await asyncio.to_thread(
            self._allowed_ids, parsed.metadata_filter_must
        )
        if candidate_scope is not None:
            allowed &= candidate_scope
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
            matched_must: dict[str, bool] = {}
            must_evidence: dict[str, str] = {}
            for condition in parsed.metadata_filter_must:
                matched, detail = _candidate_condition_match(
                    metadata, profile, condition
                )
                condition_key = (
                    f"{condition.field} {condition.operator} {condition.value}"
                )
                matched_must[condition_key] = matched
                must_evidence[condition_key] = detail

            matched_should: list[str] = []
            for condition in parsed.metadata_filter_should:
                matched, detail = _candidate_condition_match(
                    metadata, profile, condition
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
                    must_evidence=must_evidence,
                    matched_should=matched_should,
                    metadata=metadata,
                    profile=profile,
                )
            )
        return sorted(results, key=lambda item: item.final_score, reverse=True)[:limit]
