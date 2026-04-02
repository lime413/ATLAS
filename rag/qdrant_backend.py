"""Qdrant: remote (Docker) by default, or embedded ``qdrant_store/`` for faster bulk indexing (more RAM)."""

from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient

# Published by docker-compose.qdrant.yml
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def use_embedded_qdrant() -> bool:
    """Embedded mode: ``ATLAS_QDRANT_EMBEDDED=1`` (or ``true`` / ``yes`` / ``on``)."""
    v = os.environ.get("ATLAS_QDRANT_EMBEDDED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def resolve_qdrant_url(url: str | None = None) -> str:
    """explicit *url* → ``ATLAS_QDRANT_URL`` → :data:`DEFAULT_QDRANT_URL`."""
    resolved = (url or os.environ.get("ATLAS_QDRANT_URL") or DEFAULT_QDRANT_URL).strip()
    if not resolved:
        raise ValueError("Qdrant URL is empty; set --qdrant-url or ATLAS_QDRANT_URL.")
    return resolved


def verify_qdrant_reachable(client: QdrantClient) -> None:
    """Fail fast before a long index build."""
    try:
        client.get_collections()
    except Exception as e:
        if use_embedded_qdrant():
            raise RuntimeError(
                "Не удаётся открыть локальный Qdrant (qdrant_store). Проверьте путь и права на каталог."
            ) from e
        raise RuntimeError(
            "Не удаётся связаться с Qdrant. Поднимите сервер, например: "
            "docker compose -f docker-compose.qdrant.yml up -d"
        ) from e


def open_qdrant_client(url: str | None = None, *, index_dir: Path | None = None) -> QdrantClient:
    """
    - **Embedded** (``ATLAS_QDRANT_EMBEDDED=1``): ``QdrantClient(path=index_dir/qdrant_store)`` — быстрее
      массовый upsert, выше RAM (как раньше «на ночь»).
    - **Иначе**: HTTP‑сервер; по умолчанию gRPC к тому же хосту (меньше оверхеда, чем REST).
    """
    if use_embedded_qdrant():
        if index_dir is None:
            raise ValueError("Embedded Qdrant requires index_dir=…")
        return QdrantClient(path=str(index_dir / "qdrant_store"))

    resolved = resolve_qdrant_url(url)
    api_key = os.environ.get("ATLAS_QDRANT_API_KEY") or None
    prefer = os.environ.get("ATLAS_QDRANT_PREFER_GRPC", "1").strip().lower() not in ("0", "false", "no")
    return QdrantClient(
        url=resolved,
        api_key=api_key,
        prefer_grpc=prefer,
        check_compatibility=False,
    )
