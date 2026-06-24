from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cleaning import clean_profile
from .config import Settings
from .documents import profile_to_document
from .embeddings import build_faiss_index
from .storage import connect, replace_candidates


@dataclass
class PipelineResult:
    source_records: int
    stored_records: int
    records_with_warnings: int
    warning_count: int
    cleaned_jsonl_path: Path
    database_path: Path
    embedding_result: dict[str, int | str] | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"line {line_number}: {error}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: expected JSON object")
                continue
            records.append(value)
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Invalid JSONL records ({len(errors)}):\n{preview}")
    return records


def run_pipeline(
    source_path: Path,
    cleaned_jsonl_path: Path,
    database_path: Path,
    index_dir: Path,
    settings: Settings,
    skip_embeddings: bool = False,
) -> PipelineResult:
    raw_records = read_jsonl(source_path)
    seen_ids: set[str] = set()
    prepared = []
    records_with_warnings = 0
    warning_count = 0

    for raw_profile in raw_records:
        clean, warnings = clean_profile(raw_profile)
        candidate_id = str(clean["user_id"])
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        document = profile_to_document(clean)
        content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
        prepared.append((raw_profile, clean, document, content_hash, warnings))
        if warnings:
            records_with_warnings += 1
            warning_count += len(warnings)

    cleaned_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_jsonl_path.write_text(
        "".join(
            json.dumps(clean_profile, ensure_ascii=False, separators=(",", ":")) + "\n"
            for _, clean_profile, _, _, _ in prepared
        ),
        encoding="utf-8",
    )

    connection = connect(database_path)
    try:
        stored_records = replace_candidates(connection, prepared)
        embedding_result = (
            None
            if skip_embeddings
            else build_faiss_index(connection, index_dir, settings)
        )
    finally:
        connection.close()

    return PipelineResult(
        source_records=len(raw_records),
        stored_records=stored_records,
        records_with_warnings=records_with_warnings,
        warning_count=warning_count,
        cleaned_jsonl_path=cleaned_jsonl_path,
        database_path=database_path,
        embedding_result=embedding_result,
    )
