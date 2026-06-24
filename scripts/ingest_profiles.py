from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from backend.app.ingestion.config import Settings
from backend.app.ingestion.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and index candidate profiles")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("1000_Desensitization_profiles.jsonl"),
    )
    parser.add_argument(
        "--cleaned-jsonl",
        type=Path,
        default=Path("data/processed/candidates.cleaned.jsonl"),
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
        "--skip-embeddings",
        action="store_true",
        help="Build cleaned storage and BM25 only; useful before an API key is configured.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        source_path=args.source,
        cleaned_jsonl_path=args.cleaned_jsonl,
        database_path=args.database,
        index_dir=args.index_dir,
        settings=Settings.from_env(),
        skip_embeddings=args.skip_embeddings,
    )
    payload = asdict(result)
    payload["database_path"] = str(payload["database_path"])
    payload["cleaned_jsonl_path"] = str(payload["cleaned_jsonl_path"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
