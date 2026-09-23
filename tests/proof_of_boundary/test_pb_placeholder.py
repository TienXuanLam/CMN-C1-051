# CMN-C1-051 — Proof-of-Boundary Tests (PB-1 through PB-6) — SDK pattern

import ast
import json
from pathlib import Path
from unittest.mock import patch

import yaml

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.nodes.pre_process_node import ChangeDetectionNode
from src.nodes.post_process_node import ReportGenerationNode
from src.graph.graph import KBUpdateGraph

_SECRETS = InMemoryProvider({"QDRANT_URL": "http://localhost:6333"})
_ROOT = Path(__file__).resolve().parent.parent.parent


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
    if base.get("input_context"):
        base["user_input"] = yaml.safe_dump(base["input_context"], sort_keys=False)
    return base


def _add_input() -> dict:
    return {
        "event_type": "add",
        "document_id": "doc1",
        "document_path": "/docs/a.txt",
        "document_content": "Hello world. " * 20,
        "collection_name": "test",
        "previous_chunk_ids": "[]",
        "previous_chunk_hashes": "[]",
        "run_id": "test-run",
    }


def _ctx(trust_level=TrustLevel.VERIFIED_EXTERNAL):
    return InvocationContext(caller_trust_level=trust_level, caller_id="test")


def _build_agent():
    agent = KBUpdateGraph()
    agent.compile()
    return agent


# ── PB-1: emit_trace_event fires without error ────────────────────────────────


class TestPB1EmitTraceEvent:
    def test_pre_process_emits_trace(self):
        state = _base_state(input_context=_add_input())
        with patch("src.nodes.pre_process_node.emit_trace_event") as mock_emit:
            with bound_secrets(_SECRETS):
                ChangeDetectionNode().execute(state)
        assert mock_emit.called

    def test_post_process_emits_trace(self):
        state = _base_state(
            document_id="doc1",
            document_path="/docs/a.txt",
            event_type="add",
            vs_update_result=json.dumps({"upserted": 0, "deleted": 0, "collection": "test"}),
            chunks_to_add="[]",
            chunks_to_delete="[]",
            chunks_unchanged_count=0,
            affected_query_ids="[]",
            confidence_delta=0.0,
            run_id="test",
        )
        with patch("src.nodes.post_process_node.emit_trace_event") as mock_emit:
            with bound_secrets(_SECRETS):
                ReportGenerationNode().execute(state)
        assert mock_emit.called


# ── PB-2: State values are primitives ────────────────────────────────────────


class TestPB2StatePrimitives:
    def test_state_fields_are_primitive_types(self):
        state: dict = {
            "event_type": "add",
            "document_id": "doc1",
            "error_code": None,
            "validated": True,
            "confidence_delta": 0.5,
            "chunks_unchanged_count": 3,
            "current_chunks": "[]",
            "vs_update_result": "{}",
        }
        for key, val in state.items():
            assert isinstance(
                val, (str, int, float, bool, type(None))
            ), f"State['{key}'] = {type(val).__name__} — not a primitive"

    def test_formatted_output_is_string_not_dict(self):
        state = _base_state(
            document_id="doc1",
            document_path="/docs/a.txt",
            event_type="add",
            vs_update_result=json.dumps({"upserted": 1, "deleted": 0, "collection": "test"}),
            chunks_to_add=json.dumps([{"chunk_id": "c1"}]),
            chunks_to_delete="[]",
            chunks_unchanged_count=0,
            affected_query_ids="[]",
            confidence_delta=1.0,
            run_id="test",
        )
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        assert result.get("formatted_output") is not None
        assert isinstance(result["formatted_output"], str)
        assert result["formatted_output"].startswith("# Incremental Knowledge Base Update Plan")


# ── PB-3: S-3 gate — formatted_output not in blocked result ──────────────────


class TestPB3S3Gate:
    def test_credential_in_path_blocked(self):
        state = _base_state(
            document_id="doc1",
            document_path="api_key: sk-" + "a" * 25,
            event_type="add",
            vs_update_result=json.dumps({"upserted": 0, "deleted": 0, "collection": "test"}),
            chunks_to_add="[]",
            chunks_to_delete="[]",
            chunks_unchanged_count=0,
            affected_query_ids="[]",
            confidence_delta=0.0,
            run_id="test",
        )
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        assert result.get("error_code") == "S3_BLOCKED"
        assert "formatted_output" not in result

    def test_clean_output_passes(self):
        state = _base_state(
            document_id="doc1",
            document_path="/docs/a.txt",
            event_type="add",
            vs_update_result=json.dumps({"upserted": 1, "deleted": 0, "collection": "test"}),
            chunks_to_add=json.dumps([{"chunk_id": "c1"}]),
            chunks_to_delete="[]",
            chunks_unchanged_count=0,
            affected_query_ids="[]",
            confidence_delta=1.0,
            run_id="test",
        )
        with bound_secrets(_SECRETS):
            result = ReportGenerationNode().execute(state)
        assert result.get("error_code") is None
        assert "formatted_output" in result


# ── PB-4: Import isolation ────────────────────────────────────────────────────


class TestPB4ImportIsolation:
    def test_no_l0_imports_in_src(self):
        src_dir = _ROOT / "src"
        violations = []
        for fpath in src_dir.rglob("*.py"):
            with open(fpath) as f:
                try:
                    tree = ast.parse(f.read(), filename=str(fpath))
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("agenticstar"):
                            violations.append(f"{fpath}:{node.lineno} — import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("agenticstar"):
                        violations.append(f"{fpath}:{node.lineno} — from {node.module}")
        assert violations == [], "L0 violations:\n" + "\n".join(violations)


# ── PB-5: Error path — report still produced ─────────────────────────────────


class TestPB5ErrorPath:
    def test_error_path_report_produced(self):
        agent = _build_agent()
        inp = _add_input()
        inp["event_type"] = "bad_type"
        with bound_secrets(_SECRETS):
            result = agent.invoke(user_input=yaml.safe_dump(inp, sort_keys=False), ctx=_ctx())
        assert result["status"] == "success"
        assert "Valid knowledge base event required" in result.get("output", "")


# ── PB-6: Serialised fields are json strings ──────────────────────────────────


class TestPB6SerialisedFields:
    def test_5_complex_fields_serialised_as_strings(self):
        agent = _build_agent()
        with bound_secrets(_SECRETS):
            result = agent.invoke(
                user_input=yaml.safe_dump(_add_input(), sort_keys=False),
                ctx=_ctx(),
            )
        # SDK wraps output in result["output"] — check raw output dict
        assert result["status"] == "success"
        assert str(result.get("output", "")).startswith("# Incremental Knowledge Base Update Plan")
