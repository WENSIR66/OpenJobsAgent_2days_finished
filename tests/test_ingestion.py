from __future__ import annotations

from backend.app.ingestion.cleaning import clean_profile
from backend.app.ingestion.documents import profile_to_document
from backend.app.ingestion.storage import bm25_search, connect, replace_candidates


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
