"""Standalone FastAPI adapter for CMN-C1-051."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory
from shared.secrets.inmemory_provider import InMemoryProvider
from src.graph.graph import KBUpdateGraph

AGENT_ID = "CMN-C1-051"

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config: dict[str, Any] = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}

_configured_secrets_provider = secrets_factory(namespace="cmn", agent_name="cmn-c1-051")
_azure_secret_keys = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)
_secrets_provider = InMemoryProvider(
    {
        key: value
        for key in _azure_secret_keys
        if (value := os.environ.get(key) or _configured_secrets_provider.get(key)) is not None
    },
    namespace="cmn",
    agent_name="cmn-c1-051",
)

agent = KBUpdateGraph(config=_config)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    agent.compile()
    agent.provision_secrets(_secrets_provider)
    yield


app = FastAPI(title="IncrementalKnowledgeBaseUpdatePlanner", lifespan=_lifespan)


class InvokeRequest(BaseModel):
    """Marketplace-compatible plain-text request contract."""

    model_config = ConfigDict(extra="forbid")

    input: str
    session_id: str = ""


def _bearer_matches(supplied: str, expected: str) -> bool:
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel,
    authorization: str,
    invoke_auth_token: str | None,
    internal_runner_token: str | None,
) -> TrustLevel:
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust_level = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )
    ctx = InvocationContext(
        session_id=req.session_id or str(uuid4()),
        caller_trust_level=trust_level,
        caller_id=getattr(request.state, "caller_id", ""),
    )
    with bound_secrets(agent._secrets_provider):
        return cast(dict[str, Any], agent.invoke(user_input=req.input, ctx=ctx))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": AGENT_ID}
