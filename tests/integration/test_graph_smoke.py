# CMN-C1-051 — Integration smoke tests

import json

import yaml

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.graph.graph import KBUpdateGraph
from src.nodes.main_node import _chunk_text, _sha256

_SECRETS = InMemoryProvider({"QDRANT_URL": "http://localhost:6333"})


def _ctx():
    return InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="test")


def _build_agent():
    agent = KBUpdateGraph()
    agent.compile()
    return agent


def _add_input(**overrides) -> dict:
    inp = {
        "event_type": "add",
        "document_id": "doc1",
        "document_path": "/docs/a.txt",
        "document_content": "Hello world. " * 30,
        "collection_name": "test",
        "previous_chunk_ids": "[]",
        "previous_chunk_hashes": "[]",
        "run_id": "smoke-test",
    }
    inp.update(overrides)
    return inp


def test_graph_compiles():
    agent = _build_agent()
    assert agent._compiled is not None


def test_state_schema():
    from src.schemas.state import KBUpdateState

    assert KBUpdateGraph().state_schema is KBUpdateState


def test_shim_agent_py_is_alias():
    # Compatibility shim (src/agent.py) — kept intentionally for callers that
    # still import KBUpdateAgent. Canonical class is KBUpdateGraph.
    from src.agent import KBUpdateAgent

    assert KBUpdateAgent is KBUpdateGraph


def test_full_pipeline_add():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(_add_input(), sort_keys=False),
            ctx=_ctx(),
        )
    assert result["status"] == "success", f"Pipeline failed: {result}"


def test_pipeline_delete():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(
                _add_input(
                    event_type="delete",
                    previous_chunk_ids=json.dumps(["doc1_chunk_0", "doc1_chunk_1"]),
                ),
                sort_keys=False,
            ),
            ctx=_ctx(),
        )
    assert result["status"] == "success"


def test_pipeline_s1_invalid_event():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(_add_input(event_type="bad_type"), sort_keys=False),
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert "Valid knowledge base event required" in result.get("output", "")


def test_s1_validation_returns_guidance_without_llm_work():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(_add_input(event_type="bad_type"), sort_keys=False),
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert "MainNode" in result.get("node_history", [])
    assert "ReportGenerationNode" in result.get("node_history", [])
    assert "event_type" in result.get("output", "")


def test_pipeline_s1_missing_content():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(_add_input(document_content=""), sort_keys=False),
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert "document_content" in result.get("output", "")


def test_pipeline_s2_pii_is_safely_masked_by_framework():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(
                _add_input(document_content="contact user@example.com for details"), sort_keys=False
            ),
            ctx=_ctx(),
        )
    assert result["status"] == "success"
    assert "user@example.com" not in str(result.get("output", ""))


def test_pipeline_s2_credential_blocked():
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(_add_input(document_content="api_key: sk-" + "a" * 25), sort_keys=False),
            ctx=_ctx(),
        )
    assert result["status"] == "error"


def test_pipeline_update_with_prior():
    content = "same content. " * 20
    raw = _chunk_text(content, 512, 64)
    agent = _build_agent()
    with bound_secrets(_SECRETS):
        result = agent.invoke(
            user_input=yaml.safe_dump(
                _add_input(
                    event_type="update",
                    document_content=content,
                    previous_chunk_hashes=json.dumps([_sha256(c) for c in raw]),
                    previous_chunk_ids=json.dumps([f"doc1_chunk_{i}" for i in range(len(raw))]),
                ),
                sort_keys=False,
            ),
            ctx=_ctx(),
        )
    assert result["status"] == "success"
