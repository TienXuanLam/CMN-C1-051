# CMN-C1-051 Design

**L1 Base:** `AgentBaseGraph`

## Runtime contract

- Graph: `src.graph.graph.KBUpdateGraph`
- Trust: `VERIFIED_EXTERNAL`
- Input: one plain-text JSON or YAML string in `user_input`
- Output: Markdown
- SDK: `agenticstar-agentcore[marketplace,openai,platform-rag,platform-memory,platform-db,platform-storage-azure]==1.0.3`

Example input:

```yaml
event_type: add
document_id: onboarding-v1
document_path: docs/onboarding.md
document_content: |
  Welcome to the engineering onboarding guide.
previous_chunk_ids: []
previous_chunk_hashes: []
```

## Pipeline

1. `ChangeDetectionNode` parses and validates the plain-text event, applies size and security checks, and fixes the collection to the deployment-owned value.
2. `MainNode` calculates chunk hashes and the incremental add/delete/unchanged plan. Only sanitized counts and identifiers are sent to Azure OpenAI for an impact review.
3. `ReportGenerationNode` creates Markdown and applies the S-3 output gate.

The Azure client is created per invocation by `AzureOpenAIService` using:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`

Secrets never enter graph state, prompts, output, or progress metadata.

## Execution boundary

This release is an update planner. The in-process store supports deterministic regression tests only and is not a persistent Qdrant integration. The output explicitly instructs operators to apply the plan through an approved persistent vector-store writer. `QDRANT_URL` is therefore not declared as a runtime secret.

## Runtime configuration

`config/config.yaml` owns chunking, collection, size limit, retry/timeout, output format, HITL, and LLM tuning. Callers cannot override `collection_name`, `max_input_chars`, or `enable_impact_assessment` through the input payload.
