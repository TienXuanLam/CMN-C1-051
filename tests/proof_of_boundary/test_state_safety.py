# State Safety — CMN-C1-051

import ast
import importlib
import os
import re
import pytest

_CREDENTIAL_RE = re.compile(r"(?:^|_)(api_key|secret|password|credential|jwt|bearer|token)(?:_|$)", re.IGNORECASE)
_PROHIBITED = ["BaseModel", "InvocationContext", "dict"]  # bare dict violates msgpack-safe rule


def _state_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "schemas", "state.py"))


def _parse():
    with open(_state_path()) as f:
        return ast.parse(f.read())


def _get_class(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KBUpdateState":
            return node
    pytest.fail("KBUpdateState not found in src/schemas/state.py")


def _fields(cls):
    return [
        (item.target.id, item)
        for item in cls.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]


class TestNoCredentialFields:
    def test_no_credential_names(self):
        cls = _get_class(_parse())
        violations = [f"  {n}" for n, _ in _fields(cls) if _CREDENTIAL_RE.search(n)]
        assert not violations, "Credential-like field names:\n" + "\n".join(violations)


class TestMsgpackSafe:
    def test_no_prohibited_types(self):
        cls = _get_class(_parse())
        violations = []
        for name, item in _fields(cls):
            if item.annotation:
                names = {n.id for n in ast.walk(item.annotation) if isinstance(n, ast.Name)}
                for p in _PROHIBITED:
                    if p in names:
                        violations.append(f"  {name} uses {p}")
        assert not violations, "Prohibited types:\n" + "\n".join(violations)


class TestRequiredFields:
    _REQUIRED = {
        # Input
        "event_type",
        "document_id",
        "document_path",
        "document_content",
        "collection_name",
        "previous_chunk_hashes",
        "previous_chunk_ids",
        "max_input_chars",
        "enable_impact_assessment",
        "run_id",
        # Pipeline
        "validated",
        "change_type",
        "current_chunks",
        "chunks_to_add",
        "chunks_to_delete",
        "chunks_unchanged_count",
        "vs_update_result",
        "affected_query_ids",
        "confidence_delta",
        # Output
        "report_summary",
        "report",
        "formatted_output",
        # Errors
        "error_code",
        "error_message",
        # Audit
        "trace_events",
    }

    def test_all_required_fields(self):
        cls = _get_class(_parse())
        declared = {n for n, _ in _fields(cls)}
        missing = self._REQUIRED - declared
        assert not missing, f"Missing fields: {sorted(missing)}"

    def test_5_complex_fields_are_optional_str(self):
        """5 complex fields must be Optional[str] (json.dumps) not list/dict."""
        cls = _get_class(_parse())
        complex_fields = {
            "previous_chunk_ids",
            "chunks_to_delete",
            "chunks_to_add",
            "current_chunks",
            "vs_update_result",
        }
        for name, item in _fields(cls):
            if name not in complex_fields:
                continue
            if item.annotation:
                names = {n.id for n in ast.walk(item.annotation) if isinstance(n, ast.Name)}
                for bad in ("list", "dict"):
                    assert bad not in names, (
                        f"Field '{name}' uses bare '{bad}' — must be Optional[str] " "(json.dumps for msgpack safety)"
                    )


class TestRuntimeSafety:
    def test_state_is_not_pydantic(self):
        mod = importlib.import_module("src.schemas.state")
        cls = getattr(mod, "KBUpdateState")
        assert hasattr(cls, "__annotations__")
        try:
            from pydantic import BaseModel

            assert not issubclass(cls, BaseModel)
        except ImportError:
            pass

    def test_state_inherits_from_agent_state(self):
        mod = importlib.import_module("src.schemas.state")
        cls = getattr(mod, "KBUpdateState")
        from framework.schemas.agent_state import AgentState

        missing = set(AgentState.__annotations__) - set(cls.__annotations__)
        assert not missing, f"Missing AgentState fields: {sorted(missing)}"

    def test_formatted_output_is_optional_str(self):
        mod = importlib.import_module("src.schemas.state")
        cls = getattr(mod, "KBUpdateState")
        hints = cls.__annotations__
        assert "formatted_output" in hints, "formatted_output must be declared in state"
        hint = hints["formatted_output"]
        args = getattr(hint, "__args__", (hint,))
        for arg in args:
            if arg is type(None):
                continue
            assert arg is str, f"formatted_output must be Optional[str], not Optional[{arg}]"
