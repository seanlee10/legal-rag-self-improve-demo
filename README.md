# Legal RAG Agent + Cursor Tracing

This repository includes:

1. A LangGraph legal RAG agent that answers legal questions against a Victorian bench book in OpenSearch.
2. Cursor IDE hook-based tracing to [Arize Phoenix](https://github.com/Arize-ai/phoenix) and/or [Arize AX](https://arize.com).

<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>

## Legal RAG Agent

### Architecture

The graph runs four nodes in sequence:

1. **rewrite_query** — Extract concise legal search terms from the user's question using GPT-4o-mini.
2. **retrieve** — Hybrid search (BM25 + kNN) against OpenSearch, merged with Reciprocal Rank Fusion (RRF) and deduplicated by source.
3. **rerank** — Rerank retrieved documents using Cohere rerank-v3.5.
4. **call_model** — Generate an answer grounded in retrieved passages using GPT-4o-mini.

Core logic is in [`src/agent/graph.py`](./src/agent/graph.py).

### Prerequisites

- Python 3.10+
- An OpenSearch instance with the bench book indexed
- An OpenAI API key
- A Cohere API key (for reranking)

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
COHERE_API_KEY=your-cohere-api-key
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

## Cursor -> Phoenix / Arize AX Tracing

Trace Cursor IDE agent activity to **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** (self-hosted) and/or **[Arize AX](https://arize.com)** (cloud).

- **Phoenix:** bash, jq, and curl only (no Node or Python).
- **Arize AX:** same plus Python 3 and `opentelemetry-proto`, `grpcio` (see below).
- **Session** = one Cursor conversation (one chat); **Trace** = one message turn.
- **Shell and MCP** = one span per tool call with both input and output (disk-state merge).

### Tracing Requirements

- **bash** (macOS/Linux), **jq**, **curl**
- **Phoenix:** a running Phoenix instance (for example, `http://localhost:6006`)
- **Arize AX (optional):** Python 3 with `opentelemetry-proto` and `grpcio` (see [`.cursor/hooks/scripts/requirements.txt`](.cursor/hooks/scripts/requirements.txt))

### Tracing Installation

1. This repo already contains `.cursor/hooks.json` and `.cursor/hooks/hook-handler.sh`. If you copy this setup elsewhere, ensure:
   - `.cursor/hooks.json` exists and points to your hook script
   - `.cursor/hooks/hook-handler.sh` and `.cursor/hooks/lib/common.sh` are present
2. Configure environment (one of):
   - **Project:** Create a `.env` in the project root (see `.env.example`).
   - **Shell:** Export variables in `~/.zshrc` / `~/.bashrc`.
   - **Global:** Optional `~/.cursor-phoenix.env` is loaded if present.
3. Set at least one backend:
   - **Phoenix:** `PHOENIX_ENDPOINT` (for example, `http://localhost:6006`), optional `PHOENIX_PROJECT` (default `cursor`)
   - **Arize AX:** `ARIZE_API_KEY` and `ARIZE_SPACE_ID`, and install deps:
     `python3 -m venv .cursor/hooks/scripts/.venv && .cursor/hooks/scripts/.venv/bin/pip install -r .cursor/hooks/scripts/requirements.txt`
   - Enable tracing: `CURSOR_TRACE_PHOENIX=true` or `ARIZE_TRACE_ENABLED=true`

### Hooks Supported (12)

| Hook | What is sent |
|------|----------------|
| beforeSubmitPrompt | User prompt span (root of trace) |
| afterAgentResponse | Agent response span |
| afterAgentThought | Agent thinking span |
| beforeShellExecution | State only (for merge) |
| afterShellExecution | One "Shell" span with command + output (merged) |
| beforeMCPExecution | State only (for merge) |
| afterMCPExecution | One "MCP: {tool}" span with input + result (merged) |
| beforeReadFile | Read file span |
| afterFileEdit | File edit span |
| stop | Agent stop span; cleanup state for this turn |
| beforeTabFileRead | Tab read file span |
| afterTabFileEdit | Tab file edit span |

Every span includes `conversation_id` (session) and `generation_id` (trace) in attributes so Phoenix can group by conversation and by turn.

### How Shell/MCP Merge Works

- **before** hooks do not create a span; they push a small JSON payload (for example, command, cwd, start time) to a disk-backed FIFO stack keyed by `generation_id` (and tool name for MCP).
- **after** hooks pop from the stack and create one span with start time from state and output from the event.
- On **stop**, state for that `generation_id` is removed so the state directory does not grow.

### Tracing Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| PHOENIX_ENDPOINT | `http://localhost:6006` | Phoenix API base URL |
| PHOENIX_PROJECT | `cursor` | Project name in Phoenix / Arize |
| PHOENIX_API_KEY | — | Optional Bearer token if Phoenix requires auth |
| ARIZE_API_KEY | — | Arize AX API key (required for Arize AX) |
| ARIZE_SPACE_ID | — | Arize AX space ID (required for Arize AX) |
| ARIZE_PROJECT_NAME | (PHOENIX_PROJECT) | Project name in Arize AX |
| CURSOR_TRACE_PHOENIX | `true` | Enable tracing |
| ARIZE_TRACE_ENABLED | `true` | Same as above |
| CURSOR_PHOENIX_STATE_DIR | `~/.cursor/cursor-phoenix-state` | State directory for merge |
| ARIZE_VERBOSE | `false` | Log to stderr |

### Verify Tracing

- **Phoenix:** Start Phoenix (`phoenix serve` or Docker), then use Cursor in this repo. In Phoenix, open project `cursor` (or your `PHOENIX_PROJECT`) and check traces grouped by `conversation_id`.
- **Arize AX:** Add `ARIZE_API_KEY` and `ARIZE_SPACE_ID` to `.env`, install the Python deps, then run [`.cursor/hooks/scripts/test_arize_ax.sh`](.cursor/hooks/scripts/test_arize_ax.sh). You should see `Export OK` and a span named `cursor-arize-ax-smoke-test`.

### Fail-open

Errors in the hook (for example, Phoenix unreachable) are logged but do not block Cursor.

## Development

```bash
make test              # unit tests
make lint              # ruff + mypy --strict
make format            # auto-fix formatting
make integration_tests # end-to-end (requires running server)
```

Local changes are hot-reloaded in LangGraph Studio. You can edit past state and rerun from previous nodes to debug.

## License

Apache-2.0
