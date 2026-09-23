"""AgentCore Platform v1.0"""

import hashlib
import json
import os
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import KBUpdateState
from src.services.azure_openai_service import AzureOpenAIService
from src.services.progress import emit_progress
from src.services.vector_store_plan import InMemoryVectorStorePlan as _MockVectorStore
from src.services.vector_store_plan import _VECTOR_STORE as _VECTOR_STORE

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


# ── MockVectorStoreInterface (real wiring pending) ────────────────────────────
# Real Qdrant wiring is pending — MockVectorStoreInterface used in all environments.
# STILL A BLOCKER for a released agent: in-process dict, no persistence across
# restarts or worker/pod replicas. The Marketplace contract therefore labels
# the result as a plan and does not declare a vector-store secret.
#
# Finding (2026-08-19, High): the prior design constructed a fresh
# _MockVectorStore() inside _vector_store_update() on every call, so every
# upsert was immediately discarded and every delete ran against an empty
# store (always deleted=0) -- confirmed by the node's own test asserting
# `deleted == 0` with the comment "mock doesn't share state between calls".
# A real vector database is an external, shared resource across
# invocations -- that is the behavior a mock standing in for it must
# reproduce. A module-level singleton is the correct (and only) way to do
# that without violating the framework's ban on mutable *node instance*
# state (self._cache-style state on a registry-cached node instance): this
# store isn't node-instance state at all, it plays the same role a real
# Qdrant client's remote database would -- external to the node, addressed
# by (collection, chunk_id), not touched by the node's own lifecycle.
#
# Finding (2026-08-19, High): the original key was f"{collection}::{chunk_id}"
# -- plain string concatenation, not a namespace-safe separator. collection=
# "tenant" + chunk_id="doc::x" produces the identical string "tenant::doc::x"
# as collection="tenant::doc" + chunk_id="x", so a delete in one collection
# could remove a record that actually belongs to a different collection
# (confirmed live: upsert into "tenant::doc"/"x", then delete from "tenant"
# with chunk_id "doc::x" removed it). Keying by the (collection, chunk_id)
# tuple itself removes the ambiguity entirely -- no separator, no collision.
def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    # Finding (2026-08-19, Medium): only the upper bound (overlap >= chunk_size)
    # was checked. A negative overlap advances `start` by MORE than chunk_size
    # each iteration (start += chunk_size - overlap), silently skipping content
    # -- confirmed live: chunk_size=4, overlap=-2 on "abcdefghij" produced
    # ["abcd", "ghij"], dropping "ef" entirely with no error.
    if overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"chunk_overlap must be < chunk_size, got overlap={overlap} >= size={chunk_size}")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MainNode(FunctionNode):
    """main slot: composite DeltaExtract → VectorStoreUpdate → ImpactAssess."""

    # ANONYMOUS: inner composite, never called by external callers directly.
    # Graph and ChangeDetectionNode enforce VERIFIED_EXTERNAL trust.
    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        llm: Any | None = None,
        llm_temperature: float = 0.2,
        llm_max_tokens: int = 1600,
        timeout_s: int = 120,
        max_retry: int = 2,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._llm = llm
        self._azure_openai = AzureOpenAIService(
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            timeout_s=timeout_s,
            max_retry=max_retry,
        )

    def execute(self, state: KBUpdateState, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        accumulated: dict[str, Any] = {}

        emit_progress("Calculating incremental knowledge-base chunk changes.", "delta_calculation")

        # Step 1: Delta extraction
        result = self._delta_extract({**state, **accumulated})
        accumulated.update(result)
        if accumulated.get("error_code"):
            return accumulated

        # Step 2: Vector store update
        result = self._vector_store_update({**state, **accumulated})
        accumulated.update(result)
        if accumulated.get("error_code"):
            return accumulated

        # Step 3: Impact assessment
        emit_progress("Generating an impact review with Azure OpenAI.", "impact_assessment")
        result = self._impact_assess({**state, **accumulated})
        accumulated.update(result)

        if "status" not in accumulated or accumulated.get("status") == AgentStatus.PENDING.value:
            accumulated["status"] = AgentStatus.SUCCESS.value

        return accumulated

    # ── Step 1: Delta extraction ──────────────────────────────────────────────

    def _delta_extract(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        change_type = state.get("change_type", "")

        # Delete path pre-populated by pre_process_node
        if change_type == "delete":
            emit_trace_event(
                event_type="delta_extract_skipped",
                payload={"reason": "delete — pre-populated by pre_process", "run_id": run_id},
                state=state,
            )
            return {}

        try:
            document_id = str(state.get("document_id", ""))
            content = str(state.get("document_content", ""))

            # Deserialise previous hashes/ids from json string. Finding
            # (2026-08-19, High): the prior comparison used prev_hashes as a
            # global set-membership test (chunk unchanged if its hash appears
            # ANYWHERE in the previous document), not keyed by chunk_id. Two
            # chunks swapping position, or a document with a repeated chunk,
            # would then keep a stale chunk_id->hash mapping in the vector
            # store unchanged even though its per-position content did
            # change. Pairing by index below (previous_chunk_ids[i] <->
            # previous_chunk_hashes[i], the same order pre_process_node
            # populates them in) and comparing each *current* chunk_id's hash
            # against its OWN previous hash fixes this: only a chunk whose
            # own id existed before AND whose hash is unchanged counts as
            # unchanged.
            prev_hashes_raw = state.get("previous_chunk_hashes", "[]")
            prev_ids_raw = state.get("previous_chunk_ids", "[]")
            prev_hashes_list: list[str] = json.loads(prev_hashes_raw) if prev_hashes_raw else []
            prev_ids: list[str] = json.loads(prev_ids_raw) if prev_ids_raw else []
            prev_hash_by_id: dict[str, str] = dict(zip(prev_ids, prev_hashes_list))

            chunk_size = int(state.get("chunk_size") or self._chunk_size)
            overlap = int(state.get("chunk_overlap") or self._chunk_overlap)

            raw_chunks = _chunk_text(content, chunk_size, overlap)
            current_chunks: list[dict[str, Any]] = [
                {"chunk_id": f"{document_id}_chunk_{i}", "text": t, "hash": _sha256(t), "index": i}
                for i, t in enumerate(raw_chunks)
            ]

            current_ids = {c["chunk_id"] for c in current_chunks}
            chunks_to_add = [c for c in current_chunks if prev_hash_by_id.get(c["chunk_id"]) != c["hash"]]
            chunks_to_delete = [cid for cid in prev_ids if cid not in current_ids]
            unchanged = len(current_chunks) - len(chunks_to_add)

            emit_trace_event(
                event_type="delta_extract_ok",
                payload={
                    "document_id": document_id,
                    "total": len(current_chunks),
                    "to_add": len(chunks_to_add),
                    "to_delete": len(chunks_to_delete),
                    "unchanged": unchanged,
                    "run_id": run_id,
                },
                state=state,
            )

            return {
                "current_chunks": json.dumps(current_chunks),
                "chunks_to_add": json.dumps(chunks_to_add),
                "chunks_to_delete": json.dumps(chunks_to_delete),
                "chunks_unchanged_count": unchanged,
            }

        except Exception as exc:  # noqa: BLE001
            emit_trace_event(
                event_type="delta_extract_error",
                payload={"error": str(exc)[:200], "run_id": run_id},
                state=state,
            )
            return {
                "error_code": "ERR_DELTA_EXTRACTION",
                "error_message": f"DeltaExtraction failed: {str(exc)[:200]}",
                "status": AgentStatus.ERROR.value,
            }

    # ── Step 2: Vector store update ───────────────────────────────────────────

    def _vector_store_update(self, state: dict[str, Any]) -> dict[str, Any]:
        # Real Qdrant wiring is pending — MockVectorStore used in all environments
        run_id = state.get("run_id", "")
        try:
            collection = str(state.get("collection_name", "default"))

            chunks_to_add = json.loads(state.get("chunks_to_add") or "[]")
            chunks_to_delete = json.loads(state.get("chunks_to_delete") or "[]")

            vs = _MockVectorStore()
            upsert_r = vs.upsert(collection, chunks_to_add) if chunks_to_add else {"upserted": 0}
            delete_r = vs.delete(collection, chunks_to_delete) if chunks_to_delete else {"deleted": 0}

            vs_result = {
                "upserted": upsert_r.get("upserted", 0),
                "deleted": delete_r.get("deleted", 0),
                "collection": collection,
            }

            emit_trace_event(
                event_type="vector_store_update_ok",
                payload={**vs_result, "run_id": run_id},
                state=state,
            )
            return {"vs_update_result": json.dumps(vs_result)}

        except Exception as exc:  # noqa: BLE001
            emit_trace_event(
                event_type="vector_store_update_error",
                payload={"error": str(exc)[:200], "run_id": run_id},
                state=state,
            )
            return {
                "error_code": "ERR_VECTOR_STORE_UPDATE",
                "error_message": f"VectorStoreUpdate failed: {str(exc)[:200]}",
                "status": AgentStatus.ERROR.value,
            }

    # ── Step 3: Impact assessment ─────────────────────────────────────────────

    def _impact_assess(self, state: dict[str, Any]) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        enable = state.get("enable_impact_assessment", True)

        if not enable:
            emit_trace_event(
                event_type="impact_assess_skipped",
                payload={"reason": "disabled via enable_impact_assessment", "run_id": run_id},
                state=state,
            )
            return {
                "affected_query_ids": json.dumps([]),
                "confidence_delta": 0.0,
                "impact_review": "Impact assessment is disabled by deployment configuration.",
            }

        # Finding (2026-08-19, High): this previously computed `deleted` from
        # len(chunks_to_delete) -- the REQUESTED delete list from
        # _delta_extract -- not the vector store's actual delete count.
        # Deleting a chunk_id that no longer exists in the store (e.g. it was
        # never upserted, or was already removed) reports vs_update_result
        # deleted=0, but confidence_delta was still computed as though the
        # deletion succeeded, producing a confidence swing for a no-op.
        # Reading from vs_update_result (the mock/real vector store's own
        # report of what it actually did) instead of the request list fixes
        # this. chunks_to_add's length is left as-is: DeltaExtract already
        # only puts a chunk in chunks_to_add when it genuinely needs
        # upserting, and _MockVectorStore.upsert() always upserts everything
        # it's given (no partial-failure case to under-report), so
        # vs_result["upserted"] and len(chunks_to_add) are equivalent for
        # this mock -- unlike delete, whose count depends on whether the
        # chunk_id was actually found in the store.
        chunks_to_add = json.loads(state.get("chunks_to_add") or "[]")
        unchanged = int(state.get("chunks_unchanged_count") or 0)
        vs_result = json.loads(state.get("vs_update_result") or "{}")

        added = len(chunks_to_add)
        deleted = int(vs_result.get("deleted", 0))
        total = added + deleted + unchanged
        confidence_delta = round(max(-1.0, min(1.0, (added - deleted) / total)), 4) if total > 0 else 0.0

        # Stub: affected_query_ids always [] until real Qdrant wiring lands
        affected_query_ids: list[str] = []

        metadata = {
            "event_type": str(state.get("event_type", "")),
            "document_id": str(state.get("document_id", "")),
            "chunks_to_add": added,
            "chunks_to_delete": deleted,
            "chunks_unchanged": unchanged,
            "confidence_delta": confidence_delta,
        }
        prompt = (
            "You are reviewing an incremental knowledge-base update plan. "
            "Explain the operational impact and give concise verification steps. "
            "Do not claim that a persistent vector-store write has occurred. "
            "Do not invent query IDs, document contents, legal conclusions, or system state. "
            f"Sanitized metadata: {json.dumps(metadata, ensure_ascii=False)}"
        )
        try:
            impact_review = self._llm_complete(prompt, state)
        except Exception as exc:  # noqa: BLE001
            emit_trace_event(
                event_type="impact_review_error",
                payload={"error_type": type(exc).__name__, "run_id": run_id},
                state=state,
            )
            return {
                "error_code": "ERR_LLM_IMPACT_REVIEW",
                "error_message": "Azure OpenAI impact review failed",
                "status": AgentStatus.ERROR.value,
            }

        emit_trace_event(
            event_type="impact_assess_ok",
            payload={
                "confidence_delta": confidence_delta,
                "affected_queries": len(affected_query_ids),
                "stub_pending_coe381": True,
                "run_id": run_id,
            },
            state=state,
        )
        return {
            "affected_query_ids": json.dumps(affected_query_ids),
            "confidence_delta": confidence_delta,
            "impact_review": impact_review,
        }

    def _llm_complete(self, prompt: str, state: dict[str, Any]) -> str:
        # STG_MOCK_MODE is the scaffold's standard Stage 5 provisional-deploy
        # toggle, set "true" by the shared deploy-stg CI job -- never in
        # production. Read directly from
        # os.environ (not ctx.secrets) since it is a CI-only structural
        # toggle, not a credential.
        if self._llm is None and os.environ.get("STG_MOCK_MODE") == "true":
            return "Review the planned chunk changes, verify retrieval quality, and apply the plan through an approved vector-store writer."
        llm = self._llm or self._azure_openai.create_client(state)
        response = llm.complete([{"role": "user", "content": prompt}])
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        return str(content).strip()
