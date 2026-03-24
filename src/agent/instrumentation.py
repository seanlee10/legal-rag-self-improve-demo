"""Arize AX tracing instrumentation for the LangGraph agent."""

import os

from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    api_key=os.getenv("ARIZE_API_KEY"),
    project_name=os.getenv("ARIZE_PROJECT_NAME", "ax-demo"),
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
