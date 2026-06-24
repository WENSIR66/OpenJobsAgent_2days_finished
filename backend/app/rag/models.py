from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MetadataCondition(BaseModel):
    field: Literal[
        "is_working",
        "current_title",
        "current_department",
        "management_level",
        "is_decision_maker",
        "total_experience_months",
        "total_experience_years",
        "highest_degree_level",
        "skills",
        "roles",
        "levels",
        "industries",
        "companies",
        "locations",
        "majors",
        "education_countries",
    ]
    operator: Literal["eq", "contains", "gte", "lte", "in"]
    value: Any


class ParsedQuery(BaseModel):
    semantic_query: str
    metadata_filter_must: list[MetadataCondition] = Field(default_factory=list)
    metadata_filter_should: list[MetadataCondition] = Field(default_factory=list)


class CandidateScore(BaseModel):
    candidate_id: str
    vector_score: float
    bm25_score: float
    metadata_should_score: float
    final_score: float
    matched_must: dict[str, bool] = Field(default_factory=dict)
    matched_should: list[str] = Field(default_factory=list)
    metadata: dict[str, Any]
    profile: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    parsed_query: ParsedQuery
    candidates: list[CandidateScore]
    answer: str
