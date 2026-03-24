"""New LangGraph Agent.

This module defines a custom graph.
"""

import agent.instrumentation  # noqa: F401 — initialize Arize tracing before graph
from agent.graph import graph

__all__ = ["graph"]
