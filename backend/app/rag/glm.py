from __future__ import annotations

import asyncio
from typing import Any, Sequence

import httpx

from backend.app.ingestion.config import Settings

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class AsyncGLMClient:
    """Async GLM client using one reusable HTTP connection pool."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not settings.api_key:
            raise ValueError("Missing ZHIPUAI_API_KEY (or GLM_API_KEY/BIGMODEL_API_KEY)")
        self.settings = settings
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient(
            base_url=settings.api_base,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                timeout=max(
                    settings.chat_timeout_seconds,
                    settings.embedding_timeout_seconds,
                ),
                connect=settings.http_connect_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive_connections,
            ),
        )

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(5):
            try:
                response = await self.http.post(path, json=body)
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    return response.json()
                if attempt == 4:
                    response.raise_for_status()
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 4:
                    raise
            except httpx.HTTPStatusError as error:
                detail = error.response.text[:500]
                raise RuntimeError(
                    f"GLM API failed with HTTP {error.response.status_code}: {detail}"
                ) from error
            await asyncio.sleep(2**attempt)
        raise RuntimeError("GLM request exhausted retries")

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            body["response_format"] = response_format
        result = await self._post("chat/completions", body)
        return result["choices"][0]["message"]["content"]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [text[: self.settings.embedding_max_chars] for text in texts]
        result = await self._post(
            "embeddings",
            {"model": self.settings.embedding_model, "input": prepared},
        )
        data = sorted(result["data"], key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(prepared):
            raise RuntimeError(
                f"Embedding API returned {len(vectors)} vectors for {len(prepared)} inputs"
            )
        return vectors

    async def close(self) -> None:
        if self._owns_client:
            await self.http.aclose()
