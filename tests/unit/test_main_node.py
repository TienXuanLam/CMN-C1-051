# CMN-C1-051 — Unit Tests: MainNode contract + inner step logic

import json
import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.nodes.main_node import MainNode, _chunk_text, _sha256

_SECRETS = InMemoryProvider({"QDRANT_URL": "http://localhost:6333"})


def _state(**overrides) -> dict:
    base = {
        "user_input": "",
        "input_context": {},
        "caller_trust_level": "INTERNAL",
        "caller_id": "test",
        "correlation_id": "test",
        "session_id": "test",
        "thread_id": "test",
        "trace_id": "",
        "hitl_allowed": True,
        "node_history": [],
        "error_log": [],
        "status": "pending",
        "execution_time": {},
        "error_code": None,
        "error_message": None,
        "event_type": "add",
        "document_id": "doc1",
        "document_path": "/docs/a.txt",
        "document_content": "Hello world. " * 20,
        "collection_name": "test",
        "change_type": "add",
        "validated": True,
        "previous_chunk_ids": "[]",
        "previous_chunk_hashes": "[]",
        "run_id": "test-run",
        "enable_impact_assessment": True,
    }
    base.update(overrides)
    return base


class TestMainNodeContract:
    def test_required_trust_level(self):
        assert MainNode.required_trust_level == TrustLevel.ANONYMOUS

    def test_execute_returns_dict(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_state())
        assert isinstance(result, dict)

    def test_upstream_error_returns_empty(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_state(error_code="S1_MISSING_CONTENT"))
        assert result == {}

    def test_returns_success_status(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node.execute(_state())
        assert result.get("status") == AgentStatus.SUCCESS.value


class TestChunkText:
    def test_negative_overlap_rejected(self):
        # Regression: only overlap >= chunk_size was rejected. A negative
        # overlap advances `start` by MORE than chunk_size per iteration,
        # silently skipping content -- confirmed live: chunk_size=4,
        # overlap=-2 on "abcdefghij" produced ["abcd", "ghij"], dropping
        # "ef" with no error at all.
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            _chunk_text("abcdefghij", 4, -2)

    def test_zero_overlap_still_allowed(self):
        assert _chunk_text("abcdefgh", 4, 0) == ["abcd", "efgh"]


class TestDeltaExtract:
    def test_new_document_all_chunks_added(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._delta_extract(_state())
        chunks_to_add = json.loads(result["chunks_to_add"])
        assert len(chunks_to_add) > 0
        assert result["chunks_unchanged_count"] == 0

    def test_unchanged_document_no_adds(self):
        content = "same content. " * 30
        raw = _chunk_text(content, 512, 64)
        prev_hashes = json.dumps([_sha256(c) for c in raw])
        prev_ids = json.dumps([f"doc1_chunk_{i}" for i in range(len(raw))])
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._delta_extract(
                _state(
                    document_content=content,
                    previous_chunk_hashes=prev_hashes,
                    previous_chunk_ids=prev_ids,
                    change_type="update",
                )
            )
        assert json.loads(result["chunks_to_add"]) == []
        assert result["chunks_unchanged_count"] == len(raw)

    def test_delete_path_skipped(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._delta_extract(
                _state(
                    change_type="delete",
                    chunks_to_delete=json.dumps(["doc1_chunk_0"]),
                    chunks_to_add=json.dumps([]),
                )
            )
        assert "chunks_to_add" not in result
        assert "current_chunks" not in result

    def test_invalid_chunk_params_returns_error(self):
        node = MainNode(chunk_size=64, chunk_overlap=64)
        with bound_secrets(_SECRETS):
            result = node._delta_extract(_state(document_content="hello"))
        assert result.get("error_code") == "ERR_DELTA_EXTRACTION"

    def test_chunk_structure(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._delta_extract(_state())
        for chunk in json.loads(result["chunks_to_add"]):
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "hash" in chunk
            assert "index" in chunk

    def test_unchanged_count_invariant(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._delta_extract(_state())
        total = len(json.loads(result["current_chunks"]))
        added = len(json.loads(result["chunks_to_add"]))
        unchanged = result["chunks_unchanged_count"]
        assert added + unchanged == total
        assert unchanged >= 0

    def test_serialised_fields_are_strings(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._delta_extract(_state())
        for field in ["current_chunks", "chunks_to_add", "chunks_to_delete"]:
            assert isinstance(result[field], str), f"{field} must be json string"

    def test_chunk_size_change_causes_full_reindex(self):
        content = "abcdefghij" * 80
        raw = _chunk_text(content, 512, 64)
        prev_hashes = json.dumps([_sha256(c) for c in raw])
        prev_ids = json.dumps([f"doc1_chunk_{i}" for i in range(len(raw))])
        node = MainNode(chunk_size=256, chunk_overlap=32)
        with bound_secrets(_SECRETS):
            result = node._delta_extract(
                _state(
                    document_content=content,
                    previous_chunk_hashes=prev_hashes,
                    previous_chunk_ids=prev_ids,
                    change_type="update",
                )
            )
        assert result["chunks_unchanged_count"] == 0

    def test_swapped_chunk_content_at_same_position_is_flagged_changed(self):
        # Regression: the old comparison treated a chunk as unchanged if its
        # hash appeared ANYWHERE in previous_chunk_hashes (global set
        # membership), not keyed by chunk_id/position. Two chunks with
        # identical content swapping which position they occupy previously
        # produced chunks_unchanged_count == total, silently leaving the
        # vector store never re-upserted for a real position swap. The fixed
        # comparison keys strictly by chunk_id (== position), so a chunk_id
        # whose hash actually changed relative to ITS OWN previous hash must
        # be flagged, even when that hash exists elsewhere in the previous set.
        node = MainNode(chunk_size=64, chunk_overlap=0)
        # Two 64-char chunks, A and B, occupying positions 0 and 1.
        chunk_a = "a" * 64
        chunk_b = "b" * 64
        # Previous state: position 0 = A, position 1 = B.
        prev_ids = json.dumps(["doc1_chunk_0", "doc1_chunk_1"])
        prev_hashes = json.dumps([_sha256(chunk_a), _sha256(chunk_b)])
        # Current content: position 0 = B, position 1 = A (swapped).
        with bound_secrets(_SECRETS):
            result = node._delta_extract(
                _state(
                    document_content=chunk_b + chunk_a,
                    previous_chunk_hashes=prev_hashes,
                    previous_chunk_ids=prev_ids,
                    change_type="update",
                )
            )
        # Both positions actually changed content relative to their own
        # chunk_id's previous hash -- unchanged_count must be 0, not 2.
        assert result["chunks_unchanged_count"] == 0
        assert len(json.loads(result["chunks_to_add"])) == 2


class TestVectorStoreUpdate:
    def test_upsert_chunks(self):
        node = MainNode()
        chunks = json.dumps([{"chunk_id": "vsu-upsert-c1", "text": "hello", "hash": "h", "index": 0}])
        with bound_secrets(_SECRETS):
            result = node._vector_store_update(
                _state(chunks_to_add=chunks, chunks_to_delete="[]", collection_name="test-upsert")
            )
        vs = json.loads(result["vs_update_result"])
        assert vs["upserted"] == 1
        assert vs["deleted"] == 0

    def test_delete_chunks(self):
        # Regression: the mock vector store previously discarded its data
        # between calls (a fresh _MockVectorStore() was constructed inside
        # _vector_store_update() every time), so a delete always ran against
        # an empty store and always reported deleted=0 -- even for a chunk
        # that had genuinely been upserted moments before. The module-level
        # _VECTOR_STORE fix makes this test meaningful: upsert first, then
        # delete the same chunk_id, and assert it is actually removed.
        node = MainNode()
        collection = "test-delete-roundtrip"
        with bound_secrets(_SECRETS):
            upsert_result = node._vector_store_update(
                _state(
                    chunks_to_add=json.dumps([{"chunk_id": "vsu-del-c1", "text": "hello", "hash": "h", "index": 0}]),
                    chunks_to_delete="[]",
                    collection_name=collection,
                )
            )
        assert json.loads(upsert_result["vs_update_result"])["upserted"] == 1

        with bound_secrets(_SECRETS):
            result = node._vector_store_update(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete=json.dumps(["vsu-del-c1"]),
                    collection_name=collection,
                )
            )
        vs = json.loads(result["vs_update_result"])
        assert vs["deleted"] == 1

    def test_delete_nonexistent_chunk_reports_zero(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._vector_store_update(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete=json.dumps(["vsu-never-existed"]),
                    collection_name="test-delete-nonexistent",
                )
            )
        vs = json.loads(result["vs_update_result"])
        assert vs["deleted"] == 0

    def test_no_cross_collection_deletion_via_key_collision(self):
        # Regression: the old key was f"{collection}::{chunk_id}" -- plain
        # string concatenation, not a namespace-safe separator.
        # collection="tenant" + chunk_id="doc::x" produced the identical
        # string "tenant::doc::x" as collection="tenant::doc" + chunk_id="x",
        # so a delete request scoped to one collection could remove a record
        # that actually belongs to a different collection (confirmed live).
        # Keying the mock store by the (collection, chunk_id) tuple removes
        # the ambiguity: there is no separator character for a crafted
        # chunk_id to exploit.
        node = MainNode()
        with bound_secrets(_SECRETS):
            node._vector_store_update(
                _state(
                    chunks_to_add=json.dumps([{"chunk_id": "x", "text": "t", "hash": "h", "index": 0}]),
                    chunks_to_delete="[]",
                    collection_name="tenant::doc",
                )
            )
            # A delete scoped to a DIFFERENT collection, using a chunk_id
            # crafted to collide under naive string concatenation.
            result = node._vector_store_update(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete=json.dumps(["doc::x"]),
                    collection_name="tenant",
                )
            )
        vs = json.loads(result["vs_update_result"])
        assert vs["deleted"] == 0, "delete scoped to 'tenant' must not remove a record owned by 'tenant::doc'"

    def test_empty_operation(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._vector_store_update(_state(chunks_to_add="[]", chunks_to_delete="[]"))
        vs = json.loads(result["vs_update_result"])
        assert vs["upserted"] == 0
        assert vs["deleted"] == 0

    def test_vs_result_is_json_string(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._vector_store_update(_state(chunks_to_add="[]", chunks_to_delete="[]"))
        assert isinstance(result["vs_update_result"], str)


class TestImpactAssess:
    def test_disabled_returns_zero(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._impact_assess(
                _state(
                    enable_impact_assessment=False,
                    chunks_to_add="[]",
                    chunks_to_delete="[]",
                    chunks_unchanged_count=0,
                )
            )
        assert result["confidence_delta"] == 0.0
        assert json.loads(result["affected_query_ids"]) == []

    def test_all_added_positive_delta(self):
        chunks = json.dumps([{"chunk_id": f"c{i}"} for i in range(5)])
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._impact_assess(
                _state(
                    chunks_to_add=chunks,
                    chunks_to_delete="[]",
                    chunks_unchanged_count=0,
                )
            )
        assert result["confidence_delta"] == pytest.approx(1.0)

    def test_delta_clamped(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._impact_assess(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete="[]",
                    chunks_unchanged_count=0,
                )
            )
        assert -1.0 <= result["confidence_delta"] <= 1.0

    def test_affected_query_ids_is_json_string(self):
        node = MainNode()
        with bound_secrets(_SECRETS):
            result = node._impact_assess(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete="[]",
                    chunks_unchanged_count=0,
                )
            )
        assert isinstance(result["affected_query_ids"], str)
        assert json.loads(result["affected_query_ids"]) == []  # stub until real Qdrant wiring lands

    def test_delete_of_nonexistent_chunk_does_not_swing_confidence(self):
        # Regression: confidence_delta was previously computed from
        # len(chunks_to_delete) -- the REQUESTED delete list -- not from
        # vs_update_result["deleted"], the vector store's own report of what
        # it actually deleted. Requesting deletion of a chunk_id that was
        # never in the store (or already removed) reports deleted=0 in
        # vs_update_result, but the old code still treated it as a real
        # deletion and swung confidence_delta toward -1.0 for a no-op.
        node = MainNode()
        vs_result = json.dumps({"upserted": 0, "deleted": 0, "collection": "test"})
        with bound_secrets(_SECRETS):
            result = node._impact_assess(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete=json.dumps(["never-existed"]),  # requested, but not actually deleted
                    chunks_unchanged_count=0,
                    vs_update_result=vs_result,
                )
            )
        assert result["confidence_delta"] == 0.0

    def test_delete_of_real_chunk_does_swing_confidence(self):
        node = MainNode()
        vs_result = json.dumps({"upserted": 0, "deleted": 1, "collection": "test"})
        with bound_secrets(_SECRETS):
            result = node._impact_assess(
                _state(
                    chunks_to_add="[]",
                    chunks_to_delete=json.dumps(["real-chunk"]),
                    chunks_unchanged_count=0,
                    vs_update_result=vs_result,
                )
            )
        assert result["confidence_delta"] == pytest.approx(-1.0)
