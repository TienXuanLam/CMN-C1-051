"""AgentCore Platform v1.0"""

# Compatibility shim — kept intentionally for tests that import KBUpdateAgent.
# Canonical class: src/graph/graph.py::KBUpdateGraph.
# All gate logic (S-1/S-2/S-3) lives in pre_process_node and post_process_node.
from src.graph.graph import KBUpdateGraph as KBUpdateAgent

__all__ = ["KBUpdateAgent"]
