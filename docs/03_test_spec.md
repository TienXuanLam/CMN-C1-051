# CMN-C1-051 Test Specification

## Required gates

- Manifest resolves `KBUpdateGraph` in namespace `cmn`.
- Dependency pin is `agenticstar-agentcore[marketplace,openai,platform-rag,platform-memory,platform-db,platform-storage-azure]==1.0.3`.
- Marketplace CLI uses `run_agent_marketplace` with agent ID `CMN-C1-051`.
- Server accepts only `input` and `session_id`.
- JSON and YAML text inputs are accepted; object-shaped HTTP bodies are rejected.
- Graph requires `VERIFIED_EXTERNAL` trust.
- Invalid event types, missing identifiers/content, and invalid/foreign chunk IDs return actionable Markdown guidance without invoking Azure OpenAI.
- Credentials, PII, and prompt-injection patterns remain security errors.
- Deployment-owned collection and input-size limits cannot be overridden by the caller.
- Chunk add/delete/unchanged counts and confidence delta remain deterministic.
- Azure OpenAI receives sanitized metadata only and is constructed from invocation-scoped secrets.
- Output begins with `# Incremental Knowledge Base Update Plan` and is not a JSON envelope.
- Pre-process, main, and post-process emit non-terminal progress events.
- Markdown passes S-3 output scanning and never includes raw Azure secrets.

## Test modes

Unit and provisional Stage 5 tests set `STG_MOCK_MODE=true`; real local acceptance must unset it and supply all three Azure OpenAI variables. A real test passes only when the response has `status=success`, all five graph nodes in `node_history`, and an Azure OpenAI impact review in the Markdown output.

## Current boundary

Persistent vector-store write behavior is not claimed or signed off. Tests validate an incremental plan, not a Qdrant mutation.

## Invalid-input guidance

Malformed document events must return `status=success` with Markdown beginning `# Valid knowledge base event required`. Security violations and runtime/LLM/output failures must remain `status=error`.
