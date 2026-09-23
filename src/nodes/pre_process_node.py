"""AgentCore Platform v1.0"""

import json
from typing import Any, Optional

import yaml

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes._security_patterns import (
    INJECTION_PATTERNS,
    scan_for_credentials,
    scan_for_pii,
)
from src.schemas.state import KBUpdateState
from src.services.progress import emit_progress

VALID_EVENT_TYPES = {"add", "update", "delete"}
DEFAULT_MAX_INPUT_CHARS = 1_000_000


def _input_error(error_code: str, error_message: str) -> dict[str, Any]:
    """Return actionable UI guidance for an invalid document event."""
    return {
        "error_code": error_code,
        "error_message": error_message,
        "input_validation_failed": "true",
        "formatted_output": (
            "# Valid knowledge base event required\n\n"
            f"I could not process the input: {error_message}\n\n"
            "Provide one YAML or JSON document event. `event_type` must be `add`, `update`, or `delete`. "
            "The `add` and `update` events also require `document_content`.\n\n"
            "**Example**\n\n"
            "```yaml\n"
            "event_type: add\n"
            "document_id: employee-policy-001\n"
            "document_path: docs/hr/employee-policy.md\n"
            "document_content: |\n"
            "  Employees receive twenty days of annual leave each year.\n"
            "previous_chunk_ids: []\n"
            "previous_chunk_hashes: []\n"
            "```"
        ),
        "status": AgentStatus.SUCCESS.value,
    }


class ChangeDetectionNode(FunctionNode):
    """pre_process slot: S-1/S-2 input gate + event validation.

    max_input_chars / enable_impact_assessment are deployment defaults
    (config/config.yaml via KBUpdateGraph.register_nodes()) -- previously this
    node was constructed with no config at all (ChangeDetectionNode()), so a
    deployment's own configured max_input_chars/enable_impact_assessment could
    never take effect regardless of whether the standalone entry point loaded
    config.yaml.

    Finding (2026-08-19, Medium): max_input_chars was previously read from caller context
    before falling back to the deployment default
    -- so a caller could pass "max_input_chars": 999999999 in the request
    body and bypass the deployment's configured cap entirely (confirmed
    live). A deployment size limit is a security control, not a
    caller-tunable preference; it must never be sourced from caller input.
    max_input_chars is now always self._max_input_chars, full stop.
    enable_impact_assessment remains caller-overridable via _field() -- it
    is a behavior toggle, not a security boundary.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        enable_impact_assessment: bool = True,
        collection_name: str = "default",
    ) -> None:
        self._max_input_chars = max_input_chars
        self._enable_impact_assessment = enable_impact_assessment
        self._default_collection_name = collection_name

    def execute(self, state: KBUpdateState, config: dict[str, Any] | None = None) -> dict[str, Any]:
        emit_trace_event("change_detection_started", {}, state)
        if state.get("error_code"):
            return {}

        raw_input = str(state.get("user_input") or "").strip()
        if not raw_input:
            return _input_error("S1_EMPTY_INPUT", "The input must contain a YAML or JSON document event.")

        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            try:
                parsed = yaml.safe_load(raw_input)
            except yaml.YAMLError:
                parsed = None
        if not isinstance(parsed, dict):
            return _input_error("S1_INVALID_INPUT", "The input must be a YAML or JSON object.")

        input_payload: dict[str, Any] = parsed

        def _field(key: str, default: Any = None) -> Any:
            v = input_payload.get(key)
            if v is not None:
                return v
            return default

        event_type = str(_field("event_type", "")).strip().lower()
        document_id = str(_field("document_id", "")).strip()
        document_path = str(_field("document_path", "")).strip()
        document_content: Optional[str] = _field("document_content")
        collection_name = self._default_collection_name
        previous_chunk_ids_raw = _field("previous_chunk_ids", "[]")
        previous_chunk_hashes_raw = _field("previous_chunk_hashes", "[]")
        max_input_chars = self._max_input_chars
        enable_impact = self._enable_impact_assessment
        run_id = str(_field("run_id", ""))

        # ── S-1: event_type validation ────────────────────────────────────────
        if event_type not in VALID_EVENT_TYPES:
            return _input_error(
                "S1_INVALID_EVENT_TYPE",
                f"`event_type` must be one of {sorted(VALID_EVENT_TYPES)}.",
            )

        if not document_id:
            return _input_error("S1_MISSING_DOCUMENT_ID", "`document_id` must be a non-empty string.")

        if not document_path:
            return _input_error("S1_MISSING_DOCUMENT_PATH", "`document_path` must be a non-empty string.")

        # Finding (2026-08-19, Medium): document_path had NO size limit at
        # all before being concatenated into scan_text and fed to the
        # injection/PII/credential scanners below -- only document_content
        # was checked against max_input_chars, and only for "add"/"update".
        # A caller could send an oversized document_path (any event_type,
        # including "delete") and force unbounded-size regex scanning work
        # while entirely bypassing the length gate. Apply the same
        # deployment-fixed cap to document_path unconditionally.
        if len(document_path) > max_input_chars:
            return _input_error(
                "S1_INPUT_TOO_LONG",
                f"`document_path` exceeds the maximum length of {max_input_chars} characters.",
            )

        if event_type in ("add", "update"):
            if not document_content or not document_content.strip():
                return _input_error(
                    "S1_MISSING_CONTENT",
                    f"`document_content` is required when `event_type` is `{event_type}`.",
                )
            if len(document_content) > max_input_chars:
                return _input_error(
                    "S1_INPUT_TOO_LONG",
                    f"`document_content` exceeds the maximum length of {max_input_chars} characters.",
                )

        # ── S-1: injection detection ──────────────────────────────────────────
        scan_text = (document_content or "") + document_path
        for pat in INJECTION_PATTERNS:
            if pat.search(scan_text):
                return {
                    "error_code": "S1_INJECTION_DETECTED",
                    "error_message": "Prompt injection pattern detected in input",
                    "status": AgentStatus.ERROR.value,
                }

        # ── S-2: PII/credential scan ──────────────────────────────────────────
        if scan_for_pii(scan_text):
            return {
                "error_code": "S2_PII_DETECTED",
                "error_message": "PII pattern detected in document content or path",
                "status": AgentStatus.ERROR.value,
            }
        if scan_for_credentials(scan_text):
            return {
                "error_code": "S2_CREDENTIAL_DETECTED",
                "error_message": "Credential pattern detected in document content",
                "status": AgentStatus.ERROR.value,
            }

        # ── Validate + normalise previous_chunk_ids / previous_chunk_hashes ────
        # Finding (2026-08-19, High): the prior normalisation only branched on
        # isinstance(..., list) -- anything else (a caller-supplied string
        # like "not-json") fell into str(previous_chunk_ids_raw), producing
        # "not-json" as the "json string", which then blew up with an
        # UNCAUGHT json.JSONDecodeError at the delete-path json.loads() below
        # (confirmed live). It also accepted a JSON object in place of a
        # list, and a list containing non-string elements ([1, true, null]) --
        # the HTTP adapter (server.py) only checks isinstance(parsed, list),
        # never per-element types, and a caller invoking agent.invoke()
        # directly (bypassing the HTTP adapter entirely) hits this node with
        # no upstream validation at all. _parse_chunk_id_list() below is the
        # single validation point both callers rely on.
        def _parse_chunk_id_list(raw: Any, field_name: str) -> list[str] | dict[str, Any]:
            if isinstance(raw, list):
                parsed = raw
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    return _input_error(
                        "S1_INVALID_CHUNK_LIST",
                        f"`{field_name}` must be a valid JSON array of strings.",
                    )
            else:
                return _input_error(
                    "S1_INVALID_CHUNK_LIST",
                    f"`{field_name}` must be a JSON array string or a list of strings.",
                )
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                return _input_error(
                    "S1_INVALID_CHUNK_LIST",
                    f"`{field_name}` must be a JSON array of strings.",
                )
            return parsed

        prev_ids_parsed = _parse_chunk_id_list(previous_chunk_ids_raw, "previous_chunk_ids")
        if isinstance(prev_ids_parsed, dict):
            return prev_ids_parsed
        prev_hashes_parsed = _parse_chunk_id_list(previous_chunk_hashes_raw, "previous_chunk_hashes")
        if isinstance(prev_hashes_parsed, dict):
            return prev_hashes_parsed

        # Finding (2026-08-19, High): nothing verified a chunk_id in
        # previous_chunk_ids actually belongs to THIS document_id. Chunk IDs
        # are minted as f"{document_id}_chunk_{i}" (main_node.py
        # _delta_extract), so any caller who learns another document's
        # chunk_id (e.g. from a prior response) could submit a delete/update
        # request under a DIFFERENT document_id but reference the victim's
        # chunk_id in previous_chunk_ids -- confirmed live: an
        # "attacker-document" delete request naming victim's
        # "victim_chunk_0" successfully deleted it from the shared
        # collection. Reject any chunk_id that isn't namespaced under this
        # request's own document_id.
        owned_prefix = f"{document_id}_chunk_"
        foreign_ids = [cid for cid in prev_ids_parsed if not cid.startswith(owned_prefix)]
        if foreign_ids:
            return _input_error(
                "S1_FOREIGN_CHUNK_ID",
                "`previous_chunk_ids` may contain only chunk IDs belonging to the supplied `document_id`.",
            )

        previous_chunk_ids = json.dumps(prev_ids_parsed)
        previous_chunk_hashes = json.dumps(prev_hashes_parsed)

        emit_trace_event(
            event_type="change_detection_ok",
            payload={
                "event_type": event_type,
                "document_id": document_id,
                "run_id": run_id,
            },
            state=state,
        )
        emit_progress(
            "Document event validated and normalized.",
            "change_detection",
            event_type=event_type,
        )

        result: dict[str, Any] = {
            "event_type": event_type,
            "document_id": document_id,
            "document_path": document_path,
            "document_content": document_content,
            "collection_name": collection_name,
            "previous_chunk_ids": previous_chunk_ids,
            "previous_chunk_hashes": previous_chunk_hashes,
            "max_input_chars": max_input_chars,
            "enable_impact_assessment": enable_impact,
            "run_id": run_id,
            "validated": True,
            "change_type": event_type,
            "status": AgentStatus.SUCCESS.value,
        }

        # Pre-populate delete path so MainNode._delta_extract can pass through
        if event_type == "delete":
            prev_ids = json.loads(previous_chunk_ids)
            result["chunks_to_delete"] = json.dumps(prev_ids)
            result["chunks_to_add"] = json.dumps([])
            result["current_chunks"] = json.dumps([])
            result["chunks_unchanged_count"] = 0

        return result
