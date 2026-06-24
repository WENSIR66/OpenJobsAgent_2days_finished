from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    headline TEXT,
    is_working INTEGER,
    current_title TEXT,
    current_department TEXT,
    management_level TEXT,
    is_decision_maker INTEGER,
    total_experience_months INTEGER,
    total_experience_years REAL,
    highest_degree_level INTEGER,
    skills_json TEXT NOT NULL,
    roles_json TEXT NOT NULL,
    levels_json TEXT NOT NULL,
    industries_json TEXT NOT NULL,
    companies_json TEXT NOT NULL,
    locations_json TEXT NOT NULL,
    majors_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    clean_json TEXT NOT NULL,
    page_content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    cleaning_warnings_json TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS candidate_fts USING fts5(
    candidate_id UNINDEXED,
    page_content,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    candidate_id TEXT NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (candidate_id, model)
);

CREATE INDEX IF NOT EXISTS idx_candidates_working ON candidates(is_working);
CREATE INDEX IF NOT EXISTS idx_candidates_experience ON candidates(total_experience_months);
CREATE INDEX IF NOT EXISTS idx_candidates_degree ON candidates(highest_degree_level);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def _bool_to_int(value: Any) -> int | None:
    return int(value) if isinstance(value, bool) else None


def replace_candidates(
    connection: sqlite3.Connection,
    rows: Iterable[
        tuple[dict[str, Any], dict[str, Any], Document, str, list[str]]
    ],
) -> int:
    connection.execute("DELETE FROM candidate_fts")
    connection.execute("DELETE FROM candidates")
    count = 0
    for raw_profile, clean_profile, document, content_hash, warnings in rows:
        metadata = document.metadata
        candidate_id = metadata["candidate_id"]
        connection.execute(
            """
            INSERT INTO candidates (
                candidate_id, headline, is_working, current_title, current_department,
                management_level, is_decision_maker, total_experience_months,
                total_experience_years, highest_degree_level, skills_json, roles_json,
                levels_json, industries_json, companies_json, locations_json, majors_json,
                metadata_json, raw_json, clean_json, page_content, content_hash,
                cleaning_warnings_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                candidate_id,
                metadata.get("headline"),
                _bool_to_int(metadata.get("is_working")),
                metadata.get("current_title"),
                metadata.get("current_department"),
                metadata.get("management_level"),
                _bool_to_int(metadata.get("is_decision_maker")),
                metadata.get("total_experience_months"),
                metadata.get("total_experience_years"),
                metadata.get("highest_degree_level"),
                json.dumps(metadata.get("skills", []), ensure_ascii=False),
                json.dumps(metadata.get("roles", []), ensure_ascii=False),
                json.dumps(metadata.get("levels", []), ensure_ascii=False),
                json.dumps(metadata.get("industries", []), ensure_ascii=False),
                json.dumps(metadata.get("companies", []), ensure_ascii=False),
                json.dumps(metadata.get("locations", []), ensure_ascii=False),
                json.dumps(metadata.get("majors", []), ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                json.dumps(raw_profile, ensure_ascii=False),
                json.dumps(clean_profile, ensure_ascii=False),
                document.page_content,
                content_hash,
                json.dumps(warnings, ensure_ascii=False),
            ),
        )
        connection.execute(
            "INSERT INTO candidate_fts(candidate_id, page_content) VALUES (?, ?)",
            (candidate_id, document.page_content),
        )
        count += 1
    connection.execute(
        """
        DELETE FROM embedding_cache
        WHERE candidate_id NOT IN (SELECT candidate_id FROM candidates)
        """
    )
    connection.commit()
    return count


def make_fts_query(query: str) -> str:
    tokens = re.findall(r"[\w+#.-]+", query, flags=re.UNICODE)
    if not tokens:
        raise ValueError("BM25 query contains no searchable terms")
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)


def bm25_search(
    connection: sqlite3.Connection, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT candidate_id, -bm25(candidate_fts) AS bm25_score
        FROM candidate_fts
        WHERE candidate_fts MATCH ?
        ORDER BY bm25(candidate_fts)
        LIMIT ?
        """,
        (make_fts_query(query), limit),
    ).fetchall()
    return [dict(row) for row in rows]
