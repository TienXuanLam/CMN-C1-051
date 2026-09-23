# CMN-C1-051 — Unit Tests: ReportGenerationNode (post_process)

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.nodes.post_process_node import ReportGenerationNode

_SECRETS = InMemoryProvider({"QDRANT_URL": "http://localhost:6333"})


def _base_state(**overrides) -> dict:
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
    }
    base.update(overrides)
    return base


class TestReportGenerationNode:
    def _success_state(self) -> dict:
        return _base_state(
            document_id="doc1",
            document_path="/docs/a.txt",
            event_type="add",
            vs_update_result=json.dumps({"upserted": 3, "deleted": 0, "collection": "col"}),
            chunks_to_add=json.dumps([{"chunk_id": f"c{i}"} for i in range(3)]),
            chunks_to_delete=json.dumps([]),
            chunks_unchanged_count=0,
            affected_query_ids=json.dumps([]),
            confidence_delta=1.0,
            run_id="test-run",
        )

    def test_trust_level(self):
        # post_process slot is never called by external callers directly;
        # SDK TrustLevel.INTERNAL is enforced at graph + ChangeDetectionNode level.
        assert ReportGenerationNode.required_trust_level == TrustLevel.ANONYMOUS

    def test_produces_report_on_success(self):
        state = self._success_state()
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        report = json.loads(result["report"])
        assert report["status"] == "success"
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output") is not None

    def test_marks_report_status_error_when_state_carries_an_error_code(self):
        # NOTE: real graph routing never delivers this state to this node --
        # AgentBaseGraph.route() sends status=ERROR straight to finalize,
        # skipping post_process (see class docstring). This test exercises the
        # node's own error-formatting logic in isolation, not a reachable path.
        state = self._success_state()
        state["error_code"] = "S1_MISSING_CONTENT"
        state["error_message"] = "missing"
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        assert "report" in result
        report = json.loads(result["report"])
        assert report["status"] == "error"

    def test_report_is_json_string(self):
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(self._success_state())
        assert isinstance(result["report"], str)
        json.loads(result["report"])  # must be valid JSON

    def test_formatted_output_is_markdown_string(self):
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(self._success_state())
        assert isinstance(result["formatted_output"], str)
        assert result["formatted_output"].startswith("# Incremental Knowledge Base Update Plan")
        assert "## Azure OpenAI Impact Review" in result["formatted_output"]

    def test_formatted_output_contains_summary_and_execution_boundary(self):
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(self._success_state())
        assert result["report_summary"] in result["formatted_output"]
        assert "approved persistent vector-store writer" in result["formatted_output"]

    def test_s3_blocks_credential(self):
        state = self._success_state()
        state["document_path"] = "api_key: sk-" + "a" * 25
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        assert result.get("error_code") == "S3_BLOCKED"
        assert "formatted_output" not in result

    def test_report_schema_complete(self):
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(self._success_state())
        report = json.loads(result["report"])
        for key in (
            "template_id",
            "document_id",
            "document_path",
            "event_type",
            "status",
            "changes",
            "impact",
            "timestamp",
        ):
            assert key in report
        assert report["template_id"] == "CMN-C1-051"

    def test_timestamp_is_utc_string(self):
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(self._success_state())
        report = json.loads(result["report"])
        assert isinstance(report["timestamp"], str)
        assert "+00:00" in report["timestamp"]

    def test_chunks_counts_from_vs_result(self):
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(self._success_state())
        report = json.loads(result["report"])
        assert report["changes"]["chunks_added"] == 3
        assert report["changes"]["chunks_deleted"] == 0

    def test_chunks_deleted_zero_from_vs_not_overridden_by_fallback(self):
        # Regression: vs_result.get("deleted") or len(chunks_to_delete)
        # was wrong — 0 or 2 == 2, masking a successful zero-deletion result.
        # Fix: vs_result.get("deleted", len(chunks_to_delete)) returns 0 correctly.
        state = _base_state(
            document_id="doc1",
            document_path="/docs/a.txt",
            event_type="update",
            vs_update_result=json.dumps({"upserted": 1, "deleted": 0, "collection": "col"}),
            chunks_to_add=json.dumps([{"chunk_id": "c1"}]),
            chunks_to_delete=json.dumps(["old_c1", "old_c2"]),  # 2 intended, but VS reports 0
            chunks_unchanged_count=0,
            affected_query_ids=json.dumps([]),
            confidence_delta=0.5,
            run_id="test",
        )
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        report = json.loads(result["report"])
        assert (
            report["changes"]["chunks_deleted"] == 0
        ), "When VS reports deleted=0, report must show 0 — not len(chunks_to_delete)=2"
