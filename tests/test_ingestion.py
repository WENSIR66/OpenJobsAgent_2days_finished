from __future__ import annotations

import asyncio
import time
from types import MethodType

import httpx

from backend.app.ingestion.cleaning import clean_profile
from backend.app.ingestion.config import Settings
from backend.app.ingestion.documents import profile_to_document
from backend.app.ingestion.storage import bm25_search, connect, replace_candidates
from backend.app.rag.glm import AsyncGLMClient
from backend.app.rag.models import MetadataCondition
from backend.app.rag.models import ParsedQuery
from backend.app.rag.query_parser import _normalize_conditions, _rule_fallback
from backend.app.rag.retriever import (
    HybridRetriever,
    _candidate_condition_match,
    _condition_matches,
    _normalize,
    _sql_condition,
)


def sample_profile() -> dict:
    return {
        "user_id": 7,
        "headline": "  Senior   Python Engineer\u200b ",
        "summary": "Builds search systems. Show less",
        "skills": ["Python", " python ", "", "FAISS"],
        "is_working": True,
        "total_experience_duration_months": 120,
        "experience": [
            {
                "is_current": True,
                "title": "Engineer",
                "company_name": "Example",
                "company_size_range": -1,
                "order_in_profile": 1,
            }
        ],
        "education": [],
        "awards": [],
        "courses": [],
        "certifications": [],
        "publications": [],
        "patents": [],
    }


def test_clean_and_document() -> None:
    clean, warnings = clean_profile(sample_profile())
    document = profile_to_document(clean)
    assert warnings == []
    assert clean["headline"] == "Senior Python Engineer"
    assert clean["summary"] == "Builds search systems."
    assert clean["skills"] == ["Python", "FAISS"]
    assert "Skills: Python, FAISS" in document.page_content
    assert "Candidate ID" not in document.page_content
    assert document.metadata["candidate_id"] == "7"
    assert document.metadata["total_experience_years"] == 10.0


def test_sqlite_bm25(tmp_path) -> None:
    raw = sample_profile()
    clean, warnings = clean_profile(raw)
    document = profile_to_document(clean)
    connection = connect(tmp_path / "test.db")
    replace_candidates(connection, [(raw, clean, document, "hash", warnings)])
    result = bm25_search(connection, "Python", limit=5)
    assert result[0]["candidate_id"] == "7"
    count = connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
    assert count == 1
    connection.close()


def test_metadata_conditions_and_normalization() -> None:
    condition = MetadataCondition(
        field="total_experience_years", operator="gte", value=5
    )
    sql, params = _sql_condition(condition)
    assert sql == "total_experience_years >= ?"
    assert params == [5]
    assert _condition_matches({"total_experience_years": 8}, condition)

    skill = MetadataCondition(field="skills", operator="contains", value="python")
    assert _condition_matches({"skills": ["Python", "FastAPI"]}, skill)
    assert _normalize({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}


def test_frontend_backend_direction_is_removed_from_filters() -> None:
    parsed = ParsedQuery(
        semantic_query="Python backend engineer",
        metadata_filter_must=[
            MetadataCondition(
                field="total_experience_years", operator="gte", value=5
            ),
            MetadataCondition(field="skills", operator="contains", value="Python"),
            MetadataCondition(
                field="roles",
                operator="in",
                value=["backend engineer", "backend developer"],
            ),
        ],
    )
    validated = _normalize_conditions(parsed)
    assert [item.field for item in validated.metadata_filter_must] == [
        "total_experience_years",
        "skills",
    ]
    assert validated.metadata_filter_should == []
    assert "Django" in validated.metadata_filter_must[1].value


def test_keyword_condition_uses_full_page_content() -> None:
    condition = MetadataCondition(
        field="skills",
        operator="in",
        value=["Python", "Django", "Flask"],
    )
    sql, params = _sql_condition(condition)
    assert "LOWER(page_content) LIKE ?" in sql
    assert params == ["%python%", "%django%", "%flask%"]
    matched, detail = _candidate_condition_match(
        {"skills": []},
        "Experience: Built web APIs with DJANGO.",
        condition,
    )
    assert matched
    assert "Django" in detail


def test_rule_fallback_splits_core_and_preference_conditions() -> None:
    parsed = _rule_fallback(
        "寻找至少5年经验的Python后端工程师，有云平台经验优先。"
    )
    assert parsed.semantic_query == "python backend engineer cloud platform experience"
    assert [(item.field, item.operator) for item in parsed.metadata_filter_must] == [
        ("total_experience_years", "gte"),
        ("skills", "in"),
    ]
    assert parsed.metadata_filter_should[0].value == [
        "AWS",
        "GCP",
        "Azure",
        "cloud platform",
        "cloud infrastructure",
    ]


def test_bm25_is_case_insensitive(tmp_path) -> None:
    raw = sample_profile()
    clean, warnings = clean_profile(raw)
    document = profile_to_document(clean)
    connection = connect(tmp_path / "case.db")
    replace_candidates(connection, [(raw, clean, document, "hash", warnings)])
    lower = bm25_search(connection, "python", limit=5)
    upper = bm25_search(connection, "PYTHON", limit=5)
    assert [item["candidate_id"] for item in lower] == [
        item["candidate_id"] for item in upper
    ]
    connection.close()


def test_vector_and_bm25_run_in_parallel() -> None:
    retriever = object.__new__(HybridRetriever)

    def allowed(self, conditions):
        return {"7"}

    async def vector(self, query, allowed_ids, limit):
        await asyncio.sleep(0.2)
        return {}

    def bm25(self, query, allowed_ids, limit):
        time.sleep(0.2)
        return {}

    retriever._allowed_ids = MethodType(allowed, retriever)
    retriever._vector_search = MethodType(vector, retriever)
    retriever._bm25_search = MethodType(bm25, retriever)

    async def run() -> float:
        started = time.perf_counter()
        result = await retriever.search(ParsedQuery(semantic_query="python"))
        assert result == []
        return time.perf_counter() - started

    assert asyncio.run(run()) < 0.32


def test_async_glm_reuses_injected_http_client() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    async def run() -> None:
        settings = Settings.from_env()
        client = httpx.AsyncClient(
            base_url=settings.api_base,
            transport=httpx.MockTransport(handler),
        )
        glm = AsyncGLMClient(settings, http_client=client)
        try:
            assert await glm.complete([{"role": "user", "content": "test"}]) == "ok"
            assert await glm.embed(["test"]) == [[0.1, 0.2]]
            assert glm.http is client
        finally:
            await client.aclose()

    asyncio.run(run())
    assert calls == ["/api/paas/v4/chat/completions", "/api/paas/v4/embeddings"]
