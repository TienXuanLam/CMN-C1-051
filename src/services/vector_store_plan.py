"""In-process plan simulator used until an approved persistent writer exists."""

from typing import Any

_VECTOR_STORE: dict[tuple[str, str], dict[str, Any]] = {}


class InMemoryVectorStorePlan:
    """Simulate planned upserts/deletes for deterministic regression tests."""

    def upsert(self, collection: str, chunks: list[dict[str, Any]]) -> dict[str, int]:
        for chunk in chunks:
            _VECTOR_STORE[(collection, chunk["chunk_id"])] = chunk
        return {"upserted": len(chunks)}

    def delete(self, collection: str, chunk_ids: list[str]) -> dict[str, int]:
        deleted = 0
        for chunk_id in chunk_ids:
            key = (collection, chunk_id)
            if key in _VECTOR_STORE:
                del _VECTOR_STORE[key]
                deleted += 1
        return {"deleted": deleted}
