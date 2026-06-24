from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Load a simple .env file without overriding existing environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    api_base: str
    chat_model: str
    embedding_model: str
    embedding_batch_size: int
    embedding_timeout_seconds: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or Path(".env"))
        api_key = (
            os.getenv("ZHIPUAI_API_KEY")
            or os.getenv("GLM_API_KEY")
            or os.getenv("BIGMODEL_API_KEY")
        )
        return cls(
            api_key=api_key,
            api_base=os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4").rstrip("/"),
            chat_model=os.getenv("GLM_CHAT_MODEL", "glm-4.5-air"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
            embedding_batch_size=max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))),
            embedding_timeout_seconds=max(
                1, int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
            ),
        )

