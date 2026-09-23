"""Root conftest for CMN-C1-051 tests."""

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STG_MOCK_MODE", "true")

# The fleet's legacy host SDK lacks the OpenAI extra. Keep the compatibility
# shim strictly test-local; production source always imports the framework's
# real AzureOpenAIClient from the mandated module path.
try:
    from shared.services.llm.azure_openai_client import AzureOpenAIClient as _AzureOpenAIClient  # noqa: F401
except ModuleNotFoundError:
    azure_module = types.ModuleType("shared.services.llm.azure_openai_client")

    class _TestAzureOpenAIClient:
        def __init__(self, config: dict) -> None:
            self.config = config

        def complete(self, messages: list[dict]) -> dict[str, str]:
            return {"content": "Test-only Azure OpenAI impact review."}

    azure_module.AzureOpenAIClient = _TestAzureOpenAIClient
    sys.modules[azure_module.__name__] = azure_module
