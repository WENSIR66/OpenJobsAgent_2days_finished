from __future__ import annotations

import asyncio
import time
from types import MethodType

import httpx

from backend.app.ingestion.cleaning import clean_profile
from backend.app.ingestion.config import Settings
from backend.app.ingestion.documents import profile_to_document
from backend.app.ingestion.storage import bm25_search, connect, replace_candidates
from backend.app.rag.conversation import (
    ConversationStore,
    _extract_candidate_refs,
    _fallback_intent,
    resolve_candidate_refs,
)
from backend.app.rag.glm import AsyncGLMClient
from backend.app.rag.models import CandidateScore
from backend.app.rag.models import ConversationState
from backend.app.rag.models import IntentDecision
from backend.app.rag.models import SearchResponse
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
from backend.app.rag.service import CandidateRAGService


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


def test_skill_condition_uses_controlled_evidence_routes() -> None:
    condition = MetadataCondition(
        field="skills",
        operator="in",
        value=["Python", "Django", "Flask"],
    )
    sql, params = _sql_condition(condition)
    assert "json_each(skills_json)" in sql
    assert "json_extract(value, '$.description')" in sql
    assert params.count("%django%") == 5
    matched, detail = _candidate_condition_match(
        {"skills": []},
        {
            "skills": [],
            "summary": "Python expert",
            "experience": [{"description": "Built web APIs with DJANGO."}],
        },
        condition,
    )
    assert matched
    assert "experience_descriptions 字段明确命中“django”" in detail

    matched, detail = _candidate_condition_match(
        {"skills": ["django"]},
        {"skills": ["DJANGO"], "experience": []},
        condition,
    )
    assert matched
    assert "skills 字段明确命中“django”" in detail


def test_skill_does_not_borrow_from_unrelated_fields() -> None:
    condition = MetadataCondition(
        field="skills",
        operator="in",
        value=["Python", "Django"],
    )
    matched, detail = _candidate_condition_match(
        {"skills": []},
        {
            "skills": [],
            "experience": [],
            "education": [{"institution_name": "Python University"}],
            "companies": ["Django LLC"],
        },
        condition,
    )
    assert not matched
    assert "skills 字段未命中" in detail


def test_roles_route_uses_headline_and_current_history_titles() -> None:
    condition = MetadataCondition(
        field="roles",
        operator="in",
        value=["data scientist"],
    )
    matched, detail = _candidate_condition_match(
        {"roles": []},
        {
            "headline": "Senior Data Scientist",
            "skills": [],
            "experience": [],
        },
        condition,
    )
    assert matched
    assert "headline 字段" in detail


def test_industry_company_location_and_major_routes() -> None:
    profile = {
        "summary": "Worked in healthcare for Acme and served clients in Shanghai.",
        "experience": [
            {
                "company_name": "Acme",
                "company_tags": ["digital health"],
                "description": "Built hospital workflow software.",
                "address_city": "Shanghai",
            }
        ],
        "education": [
            {
                "major": "Computer Science",
                "degree_str": "MSc in Computer Science",
            }
        ],
        "courses": ["Machine Learning"],
    }
    metadata = {
        "industries": [],
        "companies": ["Acme"],
        "locations": ["Shanghai"],
        "majors": ["Computer Science"],
    }
    cases = [
        ("industries", "digital health", "company_tags"),
        ("companies", "Acme", "companies"),
        ("locations", "Shanghai", "locations"),
        ("majors", "Machine Learning", "courses"),
    ]
    for field, value, source in cases:
        matched, detail = _candidate_condition_match(
            metadata,
            profile,
            MetadataCondition(field=field, operator="contains", value=value),
        )
        assert matched
        assert f"{source} 字段" in detail


def test_certification_route_and_description_isolation() -> None:
    profile = {
        "summary": "AWS certified professional",
        "certifications": [],
        "experience": [{"description": "Designed payment systems."}],
    }
    cert_match, cert_detail = _candidate_condition_match(
        {},
        profile,
        MetadataCondition(
            field="certifications",
            operator="contains",
            value="AWS certified",
        ),
    )
    description_match, description_detail = _candidate_condition_match(
        {},
        profile,
        MetadataCondition(
            field="experience_descriptions",
            operator="contains",
            value="payment systems",
        ),
    )
    assert cert_match and "summary 字段" in cert_detail
    assert description_match and "experience_descriptions 字段" in description_detail


def test_experience_description_has_its_own_scope() -> None:
    condition = MetadataCondition(
        field="experience_descriptions",
        operator="in",
        value=["payment system", "high concurrency"],
    )
    matched, detail = _candidate_condition_match(
        {},
        {
            "skills": [],
            "summary": "Payment enthusiast",
            "experience": [
                {"description": "Designed a HIGH CONCURRENCY transaction service."}
            ],
        },
        condition,
    )
    assert matched
    assert "experience_descriptions" in detail


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


def _sample_candidate(candidate_id: str, headline: str) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        vector_score=1.0,
        bm25_score=1.0,
        metadata_should_score=0.0,
        final_score=0.7,
        metadata={"candidate_id": candidate_id, "headline": headline},
        profile={"user_id": candidate_id, "headline": headline},
    )


def test_multiturn_fallback_intents_and_candidate_refs() -> None:
    results = [
        _sample_candidate("10000001", "Python Engineer"),
        _sample_candidate("10000002", "Data Engineer"),
        _sample_candidate("10000003", "Cloud Engineer"),
    ]
    state = ConversationState(
        current_query="找 Python 工程师",
        current_results=results,
    )
    assert _fallback_intent("只看有 AWS 经验的", state).intent == "refine"
    assert _fallback_intent("重新找产品经理", state).intent == "new_search"
    compare = _fallback_intent("对比第1个和第3个", state)
    assert compare.intent == "compare"
    assert compare.candidate_refs == ["1", "3"]
    assert _extract_candidate_refs("比较候选人10000001和10000003") == [
        "10000001",
        "10000003",
    ]
    assert _extract_candidate_refs("对比第一个和第三个") == ["1", "3"]
    assert _fallback_intent("找几个产品经理", state).intent == "new_search"
    resolved = resolve_candidate_refs(["1", "10000003"], results)
    assert [item.candidate_id for item in resolved] == ["10000001", "10000003"]


def test_conversation_store_keeps_only_query_and_results() -> None:
    async def run() -> None:
        store = ConversationStore()
        conversation_id, state = await store.get_or_create("session-1")
        assert conversation_id == "session-1"
        assert state.current_query is None
        candidate = _sample_candidate("10000001", "Python Engineer")
        updated = await store.update(
            conversation_id,
            current_query="找 Python 工程师",
            current_results=[candidate],
        )
        assert updated.model_dump().keys() == {
            "current_query",
            "current_results",
        }
        await store.clear(conversation_id)
        _, cleared = await store.get_or_create(conversation_id)
        assert cleared.current_query is None
        assert cleared.current_results == []

    asyncio.run(run())


def test_chat_turn_state_flow_without_external_api() -> None:
    class FakeIntentRouter:
        async def classify(self, message, state):
            if not state.current_query:
                return IntentDecision(
                    intent="new_search",
                    rewritten_query=message,
                )
            if message.startswith("只看"):
                return IntentDecision(
                    intent="refine",
                    rewritten_query=f"{state.current_query}，并且必须有 AWS 经验",
                )
            if message.startswith("对比"):
                return IntentDecision(
                    intent="compare",
                    candidate_refs=["1", "2"],
                )
            return IntentDecision(intent="follow_up", candidate_refs=["1"])

    async def run() -> None:
        service = object.__new__(CandidateRAGService)
        service.conversations = ConversationStore()
        service.intent_router = FakeIntentRouter()
        first = _sample_candidate("10000001", "Python Engineer")
        second = _sample_candidate("10000002", "Cloud Engineer")
        search_calls: list[tuple[str, set[str] | None]] = []

        async def fake_search(query, candidate_scope=None):
            search_calls.append((query, candidate_scope))
            candidates = [first, second] if candidate_scope is None else [first]
            return SearchResponse(
                query=query,
                parsed_query=ParsedQuery(semantic_query=query),
                candidates=candidates,
                answer=f"search:{query}",
            )

        async def fake_compare(message, state, refs):
            resolved = resolve_candidate_refs(refs, state.current_results)
            return "compare:" + ",".join(item.candidate_id for item in resolved)

        async def fake_follow_up(message, state, refs):
            resolved = resolve_candidate_refs(refs, state.current_results)
            return "follow:" + ",".join(item.candidate_id for item in resolved)

        service.search = fake_search
        service._compare = fake_compare
        service._follow_up = fake_follow_up

        new_result = await service.chat_turn("找 Python 工程师")
        conversation_id = new_result.conversation_id
        assert new_result.intent == "new_search"
        assert [item.candidate_id for item in new_result.candidates] == [
            "10000001",
            "10000002",
        ]

        refined = await service.chat_turn("只看有 AWS 经验的", conversation_id)
        assert refined.intent == "refine"
        assert search_calls[-1][1] == {"10000001", "10000002"}
        assert [item.candidate_id for item in refined.candidates] == ["10000001"]

        # Restore two displayed results to exercise comparison resolution.
        await service.conversations.update(
            conversation_id,
            current_query=refined.current_query,
            current_results=[first, second],
        )
        compared = await service.chat_turn("对比第1个和第2个", conversation_id)
        assert compared.answer == "compare:10000001,10000002"

        followed = await service.chat_turn("第一个为什么排第一", conversation_id)
        assert followed.answer == "follow:10000001"

    asyncio.run(run())
