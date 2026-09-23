"""AgentCore Platform v1.0"""

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.trust_level import TrustLevel

from src.nodes.pre_process_node import ChangeDetectionNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import ReportGenerationNode
from src.schemas.state import KBUpdateState


class KBUpdateGraph(AgentBaseGraph):
    """Fixed-pipeline graph for CMN-C1-051 KBUpdateAgent.

    Slot mapping:
      pre_process  → ChangeDetectionNode   (S-1/S-2 gate + event validation)
      main         → MainNode              (DeltaExtract → VectorStoreUpdate →
                                            ImpactAssess — composite)
      post_process → ReportGenerationNode  (S-3 gate + report serialisation)

    Ordinary input-contract failures are returned as handled SUCCESS states
    with Markdown guidance. MainNode skips business work and post_process clears
    the internal validation marker. Security violations and runtime/LLM errors
    remain ERROR and route directly to finalize.

    Runtime config keys (config/config.yaml via self.config):
      chunk_size, chunk_overlap, max_input_chars, enable_impact_assessment,
      collection_name

    __pre_invoke__, _security_gate_input, _security_gate_output hooks are
    NOT overridden — SDK enforces VERIFIED_EXTERNAL trust; domain gates live in nodes.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    @property
    def name(self) -> str:
        return "cmn_c1_kb_update_agent"

    @property
    def state_schema(self) -> type:
        return KBUpdateState

    def register_nodes(self) -> None:
        super().register_nodes()
        self._nodes["pre_process"] = ChangeDetectionNode(
            max_input_chars=int(self.config.get("max_input_chars", 1_000_000)),
            enable_impact_assessment=bool(self.config.get("enable_impact_assessment", True)),
            collection_name=str(self.config.get("collection_name", "default")),
        )
        self._nodes["main"] = MainNode(
            chunk_size=int(self.config.get("chunk_size", 512)),
            chunk_overlap=int(self.config.get("chunk_overlap", 64)),
            llm_temperature=float(self.config.get("llm_temperature", 0.2)),
            llm_max_tokens=int(self.config.get("llm_max_tokens", 1600)),
            timeout_s=int(self.config.get("timeout_s", 120)),
            max_retry=int(self.config.get("max_retry", 2)),
        )
        self._nodes["post_process"] = ReportGenerationNode(
            output_format=str(self.config.get("output_format", "markdown"))
        )

    # add_edges() NOT overridden — backbone wiring belongs to the framework
