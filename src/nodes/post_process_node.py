"""AgentCore Platform v1.0"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes._security_patterns import scan_for_credentials, scan_for_pii
from src.schemas.state import KBUpdateState
from src.services.progress import emit_progress

TEMPLATE_ID = "CMN-C1-051"


class ReportGenerationNode(FunctionNode):
    """post_process slot: S-3 output gate + report serialisation.

    Ordinary input-contract failures reach this node as handled SUCCESS states
    and are returned as actionable Markdown guidance. Security violations and
    runtime/LLM failures remain ERROR and do not pass through this formatter.
    """

    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(self, output_format: str = "markdown") -> None:
        if output_format != "markdown":
            raise ValueError("CMN-C1-051 supports only output_format='markdown'")
        self._output_format = output_format

    def execute(self, state: KBUpdateState, config: dict[str, Any] | None = None) -> dict[str, Any]:
        emit_trace_event("report_generation_started", {}, state)
        if state.get("input_validation_failed"):
            emit_trace_event(
                event_type="input_guidance_returned",
                payload={"run_id": str(state.get("run_id", ""))},
                state=state,
            )
            return {
                "formatted_output": str(state.get("formatted_output") or "Invalid input."),
                "input_validation_failed": None,
                "error_code": None,
                "error_message": None,
                "status": AgentStatus.SUCCESS.value,
            }

        run_id = state.get("run_id", "")
        error_code = state.get("error_code")
        error_message = state.get("error_message")

        emit_progress(
            "Formatting the incremental update plan as Markdown.",
            "report_generation",
            output_format=self._output_format,
        )

        # Parse serialised intermediate fields
        vs_result: dict[str, Any] = {}
        if state.get("vs_update_result"):
            try:
                vs_result = json.loads(state["vs_update_result"])
            except (json.JSONDecodeError, ValueError):
                pass

        chunks_to_add: list[Any] = []
        if state.get("chunks_to_add"):
            try:
                chunks_to_add = json.loads(state["chunks_to_add"])
            except (json.JSONDecodeError, ValueError):
                pass

        chunks_to_delete: list[Any] = []
        if state.get("chunks_to_delete"):
            try:
                chunks_to_delete = json.loads(state["chunks_to_delete"])
            except (json.JSONDecodeError, ValueError):
                pass

        affected_query_ids: list[Any] = []
        if state.get("affected_query_ids"):
            try:
                affected_query_ids = json.loads(state["affected_query_ids"])
            except (json.JSONDecodeError, ValueError):
                pass

        unchanged = int(state.get("chunks_unchanged_count") or 0)
        timestamp = datetime.now(timezone.utc).isoformat()

        report_dict = {
            "template_id": TEMPLATE_ID,
            "document_id": state.get("document_id", ""),
            "document_path": state.get("document_path", ""),
            "event_type": state.get("event_type", ""),
            "status": "error" if error_code else "success",
            "changes": {
                "chunks_added": vs_result.get("upserted", len(chunks_to_add)),
                "chunks_deleted": vs_result.get("deleted", len(chunks_to_delete)),
                "chunks_unchanged": unchanged,
            },
            "impact": {
                "affected_query_count": len(affected_query_ids),
                "confidence_delta": state.get("confidence_delta", 0.0),
            },
            "error_code": error_code,
            "error_message": error_message,
            "timestamp": timestamp,
        }

        if error_code:
            summary = (
                f"KB update for document '{state.get('document_id', '')}' "
                f"({state.get('event_type', 'unknown')} event) failed: "
                f"{error_code} — {error_message}"
            )
        else:
            added = report_dict["changes"]["chunks_added"]
            deleted = report_dict["changes"]["chunks_deleted"]
            delta = report_dict["impact"]["confidence_delta"]
            affected = report_dict["impact"]["affected_query_count"]
            summary = (
                f"KB update for '{state.get('document_id', '')}' "
                f"({state.get('event_type', '')}): "
                f"{added} added, {deleted} deleted, {unchanged} unchanged. "
                f"Confidence delta: {delta:+.4f}. Affected queries: {affected}."
            )

        impact_review = str(state.get("impact_review") or "No impact review was generated.")
        markdown = "\n".join(
            [
                "# Incremental Knowledge Base Update Plan",
                "",
                "## Result",
                "",
                summary,
                "",
                "## Document Event",
                "",
                f"- Event type: `{state.get('event_type', '')}`",
                f"- Document ID: `{state.get('document_id', '')}`",
                f"- Document path: `{state.get('document_path', '')}`",
                f"- Collection: `{state.get('collection_name', '')}`",
                "",
                "## Planned Chunk Changes",
                "",
                f"- Add or update: `{report_dict['changes']['chunks_added']}`",
                f"- Delete: `{report_dict['changes']['chunks_deleted']}`",
                f"- Unchanged: `{report_dict['changes']['chunks_unchanged']}`",
                f"- Confidence delta: `{report_dict['impact']['confidence_delta']:+.4f}`",
                "",
                "## Azure OpenAI Impact Review",
                "",
                impact_review,
                "",
                "## Execution Boundary",
                "",
                "This agent produces an incremental update plan. Apply it through an approved persistent vector-store writer before treating the knowledge base as updated.",
            ]
        )

        # ── S-3 output gate ───────────────────────────────────────────────────
        # Scan only dynamic content. Static Markdown headings can contain
        # words such as "Name", which broad PII detectors may flag.
        violation = self._run_output_gate(summary + impact_review, json.dumps(report_dict))
        if violation:
            emit_trace_event(
                event_type="s3_blocked",
                payload={"reason": violation, "run_id": run_id},
                state=state,
            )
            return {
                "error_code": "S3_BLOCKED",
                "error_message": f"S-3 output gate blocked: {violation}",
                "status": AgentStatus.ERROR.value,
            }

        emit_trace_event(
            event_type="report_generation_ok",
            payload={
                "document_id": state.get("document_id", ""),
                "event_type": state.get("event_type", ""),
                "status": report_dict["status"],
                "run_id": run_id,
            },
            state=state,
        )

        return {
            "report": json.dumps(report_dict),  # msgpack-safe: dict → str
            "report_summary": summary,
            "formatted_output": markdown,
            "status": AgentStatus.SUCCESS.value,
        }

    def _run_output_gate(self, summary: str, report_json: str) -> Optional[str]:
        """S-3: scan report_summary + serialised report for credentials/PII."""
        combined = summary + report_json
        if scan_for_credentials(combined):
            return "credential pattern"
        if scan_for_pii(combined):
            return "PII pattern"
        return None
