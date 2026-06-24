from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .config import Settings


class GLMEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise ValueError("Missing ZHIPUAI_API_KEY (or GLM_API_KEY/BIGMODEL_API_KEY)")
        self.settings = settings

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [text[: self.settings.embedding_max_chars] for text in texts]
        try:
            return self._embed_request(prepared)
        except RuntimeError as error:
            if "HTTP 400" not in str(error) or len(prepared) == 1:
                raise
            midpoint = len(prepared) // 2
            return self.embed(prepared[:midpoint]) + self.embed(prepared[midpoint:])

    def _embed_request(self, texts: Sequence[str]) -> list[list[float]]:
        payload = json.dumps(
            {"model": self.settings.embedding_model, "input": list(texts)},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.api_base}/embeddings",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.settings.embedding_timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                data = sorted(body["data"], key=lambda item: item.get("index", 0))
                vectors = [item["embedding"] for item in data]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
                    )
                return vectors
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                    raise RuntimeError(
                        f"Embedding API failed with HTTP {error.code}: {detail[:500]}"
                    ) from error
            except urllib.error.URLError:
                if attempt == 4:
                    raise
            time.sleep(2**attempt)
        raise RuntimeError("Embedding request exhausted retries")


def build_faiss_index(
    connection: sqlite3.Connection,
    output_dir: Path,
    settings: Settings,
) -> dict[str, int | str]:
    client = GLMEmbeddingClient(settings)
    candidates = connection.execute(
        "SELECT candidate_id, page_content, content_hash FROM candidates ORDER BY candidate_id"
    ).fetchall()
    cached_rows = connection.execute(
        "SELECT candidate_id, content_hash, vector_json FROM embedding_cache WHERE model = ?",
        (settings.embedding_model,),
    ).fetchall()
    cache = {
        row["candidate_id"]: (row["content_hash"], json.loads(row["vector_json"]))
        for row in cached_rows
    }

    pending = [
        row
        for row in candidates
        if row["candidate_id"] not in cache
        or cache[row["candidate_id"]][0] != row["content_hash"]
    ]
    for start in range(0, len(pending), settings.embedding_batch_size):
        batch = pending[start : start + settings.embedding_batch_size]
        vectors = client.embed([row["page_content"] for row in batch])
        for row, vector in zip(batch, vectors, strict=True):
            connection.execute(
                """
                INSERT INTO embedding_cache(
                    candidate_id, model, content_hash, dimensions, vector_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(candidate_id, model) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    dimensions=excluded.dimensions,
                    vector_json=excluded.vector_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    row["candidate_id"],
                    settings.embedding_model,
                    row["content_hash"],
                    len(vector),
                    json.dumps(vector),
                ),
            )
            cache[row["candidate_id"]] = (row["content_hash"], vector)
        connection.commit()
        print(f"Embedded {min(start + len(batch), len(pending))}/{len(pending)} pending documents")

    candidate_ids = [row["candidate_id"] for row in candidates]
    matrix = np.asarray([cache[candidate_id][1] for candidate_id in candidate_ids], dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(candidate_ids):
        raise RuntimeError("Invalid embedding matrix")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "candidates.faiss"))
    (output_dir / "candidates.manifest.json").write_text(
        json.dumps(
            {
                "model": settings.embedding_model,
                "dimensions": matrix.shape[1],
                "count": len(candidate_ids),
                "candidate_ids": candidate_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": settings.embedding_model,
        "dimensions": int(matrix.shape[1]),
        "count": len(candidate_ids),
        "new_embeddings": len(pending),
    }
