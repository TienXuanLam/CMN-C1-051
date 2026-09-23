"""Regression tests for Marketplace, manifest, and Azure OpenAI wiring."""

import ast
import importlib
import tomllib
from pathlib import Path

import yaml

from src.graph.graph import KBUpdateGraph

ROOT = Path(__file__).parents[2]


def test_manifest_resolves_business_graph_and_llm_capabilities() -> None:
    manifest = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
    assert manifest["namespace"] == "cmn"
    assert manifest["version"] == "1.1.0"
    assert manifest["generation_mode"] == "llm"
    assert manifest["required_trust_level"] == "VERIFIED_EXTERNAL"
    assert manifest["requires"]["extras"] == ["openai"]
    assert manifest["requires"]["secrets"] == [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    module_name, class_name = manifest["class"].rsplit(".", 1)
    assert getattr(importlib.import_module(module_name), class_name) is KBUpdateGraph


def test_project_dependency_and_cli_target_marketplace_graph() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert (
        "agenticstar-agentcore[marketplace,openai,platform-rag,platform-memory,"
        "platform-db,platform-storage-azure]==1.0.3" in project["project"]["dependencies"]
    )
    source = (ROOT / "cli.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "from src.graph.graph import KBUpdateGraph" in source
    assert 'agent_name="CMN-C1-051"' in source
    assert 'namespace="cmn"' in source


def test_azure_client_uses_invocation_scoped_required_secrets() -> None:
    source = (ROOT / "src" / "services" / "azure_openai_service.py").read_text(encoding="utf-8")
    assert "from shared.services.llm.azure_openai_client import AzureOpenAIClient" in source
    assert "ctx = InvocationContext.from_state(state)" in source
    for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"):
        assert f'ctx.secrets.require("{key}")' in source


def test_server_and_payload_expose_plain_text_input_only() -> None:
    import src.api.server as server

    assert set(server.InvokeRequest.model_fields) == {"input", "session_id"}
    payload = __import__("json").loads((ROOT / "deploy" / "invoke_payload.json").read_text(encoding="utf-8"))
    assert set(payload) == {"input", "session_id"}
    assert isinstance(payload["input"], str)


def test_business_nodes_emit_non_terminal_progress() -> None:
    for relative in (
        "src/nodes/pre_process_node.py",
        "src/nodes/main_node.py",
        "src/nodes/post_process_node.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "emit_progress(" in source
