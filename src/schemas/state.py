"""AgentCore Platform v1.0"""

from typing import Optional

from framework.schemas.agent_state import AgentState


class KBUpdateState(AgentState):
    """Flat TypedDict state for CMN-C1-051 KBUpdateAgent.

    All complex types serialised as json.dumps() strings (msgpack safety).
    No Pydantic, dataclass, dict, or list annotations.

    Field groups:
      Input:        event_type, document_id, document_path, document_content,
                    collection_name, previous_chunk_hashes, previous_chunk_ids,
                    max_input_chars, enable_impact_assessment, run_id
      Detection:    validated, change_type
      Delta:        current_chunks, chunks_to_add, chunks_to_delete, chunks_unchanged_count
      VS Update:    vs_update_result
      Impact:       affected_query_ids, confidence_delta
      Output:       report_summary, report, formatted_output
      Error:        error_code, error_message
      Audit:        trace_events
    """

    # ---------- Input ----------
    event_type: Optional[str]  # "add" | "update" | "delete"
    document_id: Optional[str]
    document_path: Optional[str]
    document_content: Optional[str]  # raw text; None for delete
    collection_name: Optional[str]
    previous_chunk_hashes: Optional[str]  # json.dumps(list[str])
    previous_chunk_ids: Optional[str]  # json.dumps(list[str])
    max_input_chars: Optional[int]
    enable_impact_assessment: Optional[bool]
    run_id: Optional[str]

    # ---------- ChangeDetectionNode ----------
    validated: Optional[bool]
    change_type: Optional[str]

    # ---------- DeltaExtractionNode ----------
    current_chunks: Optional[str]  # json.dumps(list[dict])
    chunks_to_add: Optional[str]  # json.dumps(list[dict])
    chunks_to_delete: Optional[str]  # json.dumps(list[str])
    chunks_unchanged_count: Optional[int]

    # ---------- VectorStoreUpdateNode ----------
    vs_update_result: Optional[str]  # json.dumps(dict)

    # ---------- ImpactAssessmentNode ----------
    affected_query_ids: Optional[str]  # json.dumps(list[str])
    confidence_delta: Optional[float]
    impact_review: Optional[str]

    # ---------- ReportGenerationNode ----------
    report_summary: Optional[str]
    report: Optional[str]  # json.dumps(dict) — msgpack-safe

    # ---------- Output ----------
    formatted_output: Optional[str]  # Markdown — SDK result["output"]
    input_validation_failed: Optional[str]  # "true" while returning validation guidance

    # ---------- Error propagation ----------
    error_code: Optional[str]
    error_message: Optional[str]

    # ---------- Audit ----------
    trace_events: Optional[str]  # json.dumps(list[str])
