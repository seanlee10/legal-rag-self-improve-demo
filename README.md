# Legal RAG Agent

A LangGraph agent that answers legal questions using retrieval-augmented generation (RAG) against a Victorian bench book indexed in OpenSearch.

<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>

## Architecture

The graph runs three nodes in sequence:

1. **rewrite_query** — Extracts concise legal search terms from the user's question using GPT-4o-mini
2. **retrieve** — Hybrid search (BM25 + kNN) against OpenSearch, merged with Reciprocal Rank Fusion (RRF) and deduplicated by source
3. **call_model** — Generates an answer grounded in the retrieved passages using GPT-4o-mini

Core logic is in [`src/agent/graph.py`](./src/agent/graph.py).

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenSearch instance with the bench book indexed
- An OpenAI API key

### Install

```bash
pip install -e . "langgraph-cli[inmem]"
# Or with uv:
uv sync --dev
```

### Configure

Create a `.env` file with the required credentials:

```text
OPENAI_API_KEY=sk-...
HOST=your-opensearch-host
USERNAME=your-username
PASSWORD=your-password
INDEX=your-index-name
```

### Run

```bash
langgraph dev
```

This starts the LangGraph server with Studio UI for visual debugging.

## Development

```bash
make test              # unit tests
make lint              # ruff + mypy --strict
make format            # auto-fix formatting
make integration_tests # end-to-end (requires running server)
```

Local changes are hot-reloaded in LangGraph Studio. You can edit past state and rerun from previous nodes to debug.
