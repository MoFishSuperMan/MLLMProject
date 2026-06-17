"""Frontend-facing wrapper around the document RAG service."""

from __future__ import annotations

from pathlib import Path

from .service import RagService


class RagDemoEngine(RagService):
    """Compatibility alias for older demo imports."""

    def __init__(self, processed_root: str | Path = "data/processed") -> None:
        super().__init__(processed_root=processed_root)
