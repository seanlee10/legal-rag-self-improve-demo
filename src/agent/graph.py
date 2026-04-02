"""LangGraph RAG agent with OpenSearch retriever and GPT-4o-mini.

Retrieves relevant documents from OpenSearch, then generates a response
using GPT-4o-mini with the retrieved context.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import aiohttp
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from agent.instrumentation import tracer_provider
from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opensearchpy import AsyncOpenSearch
from pydantic import Field
from typing_extensions import TypedDict

tracer = tracer_provider.get_tracer(__name__)

# --- RAG prompt ---
RAG_TEMPLATE = """
You are an AI assistant tasked with answering questions based on provided context. Your goal is to provide accurate and relevant answers using only the information given.

Here is the context you should use to answer the question:

<context>
{context}
</context>

Now, here is the question you need to answer:

<question>
{query}
</question>

Instructions:
1. Carefully read and analyze the provided context.
2. Identify key information in the context that is relevant to the question.
3. Formulate an answer to the question using only the information from the given context.
4. If the context does not contain enough information to fully answer the question, state this clearly in your response.
5. Do not use any external knowledge or information not present in the provided context.
6. Keep your answer concise and to the point, while ensuring it fully addresses the question.

Format your response as follows:
1. Begin with a brief answer to the question.
2. Follow with a more detailed explanation, if necessary.
3. If you're quoting directly from the context, use quotation marks and indicate the quote's location in the context.

Remember, it's important to rely solely on the given context and not to introduce any external information or assumptions in your answer.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# --- Query expansion prompt ---
QUERY_EXPANSION_TEMPLATE = """Extract the key legal concepts, terms, and topics from this question. \
Output a short search query (max 30 words) using precise legal terminology that would match \
a relevant passage in a legal bench book. Do not answer the question — only output search terms.

Question: {query}

Search query:"""

query_expansion_prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_TEMPLATE)


# --- Async embedding helper ---
async def generate_embedding(input_text: str) -> List[float]:
    """Generate an embedding vector via OpenAI text-embedding-3-large."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f'Bearer {os.environ["OPENAI_API_KEY"]}',
                "Content-Type": "application/json",
            },
            json={
                "model": "text-embedding-3-large",
                "input": input_text,
                "dimensions": 1024,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            return data["data"][0]["embedding"]


# --- Embedding span for drift monitoring ---
EMBEDDING_MODEL = "text-embedding-3-large"


def log_embedding_span(
    docs: List[Document],
    model_name: str = EMBEDDING_MODEL,
    parent_span: Any = None,
) -> None:
    """Log a dedicated EMBEDDING span with vectors and text for Arize drift monitoring."""
    chunks = [
        {
            "text": doc.page_content,
            "vector": doc.metadata.get("embedding", []),
            "source": doc.metadata.get("source", "unknown"),
        }
        for doc in docs
        if doc.metadata.get("embedding")
    ]
    if not chunks:
        return

    # Build context from the parent span (retriever span)
    ctx = trace.set_span_in_context(parent_span) if parent_span else None

    with tracer.start_as_current_span(
        "document_chunk_embeddings", context=ctx
    ) as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.EMBEDDING.value,
        )
        span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, model_name)
        span.set_attribute(
            SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS,
            json.dumps({"model": model_name, "dimension": len(chunks[0]["vector"])}),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({"chunk_count": len(chunks)}))
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")

        for i, chunk in enumerate(chunks):
            span.set_attribute(f"embedding.embeddings.{i}.embedding.text", chunk["text"])
            span.set_attribute(f"embedding.embeddings.{i}.embedding.vector", chunk["vector"])

        span.set_attribute(
            SpanAttributes.METADATA,
            json.dumps({"sources": [c["source"] for c in chunks]}),
        )
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{len(chunks)} embeddings logged")


def _rrf_merge(
    bm25_resp: Dict[str, Any],
    knn_resp: Dict[str, Any],
    k: int,
    rrf_k: int = 60,
) -> List[Document]:
    """Merge two OpenSearch responses using Reciprocal Rank Fusion, with dedup."""
    scores: Dict[str, float] = {}
    hit_map: Dict[str, Dict[str, Any]] = {}

    for rank, hit in enumerate(bm25_resp["hits"]["hits"]):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        hit_map.setdefault(doc_id, hit)

    for rank, hit in enumerate(knn_resp["hits"]["hits"]):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        hit_map.setdefault(doc_id, hit)

    ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    docs: List[Document] = []
    seen_sources: set[str] = set()
    for doc_id in ranked_ids:
        if len(docs) >= k:
            break
        hit = hit_map[doc_id]
        source_data = hit["_source"]
        source_id = source_data.get("metadata", {}).get("source", doc_id)
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        metadata = {**source_data.get("metadata", {}), "score": scores[doc_id]}
        if "embedding" in source_data:
            metadata["embedding"] = source_data["embedding"]
        docs.append(
            Document(page_content=source_data["chunk"], metadata=metadata)
        )

    return docs


# --- Async OpenSearch retriever ---
class OpenSearchRetriever(BaseRetriever):
    """Retrieve documents from OpenSearch using hybrid text + kNN search."""

    k: int = 10
    client: Any = Field(description="AsyncOpenSearch client")
    index: str = Field(description="Index name to search")

    async def _aget_relevant_documents(  # type: ignore[override]
        self,
        query: str,
        *,
        run_manager: Any = None,
        search_query: str = "",
    ) -> List[Document]:
        """Search OpenSearch with separate BM25 and kNN queries fused via RRF."""
        fetch_size = self.k * 3
        bm25_text = search_query or query
        embedding = await generate_embedding(query)

        bm25_query = {
            "size": fetch_size,
            "query": {
                "multi_match": {
                    "query": bm25_text,
                    "fields": ["chunk", "metadata.title^2"],
                }
            },
        }
        knn_query = {
            "size": fetch_size,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": fetch_size,
                    }
                }
            },
        }

        bm25_resp, knn_resp = await asyncio.gather(
            self.client.search(index=self.index, body=bm25_query),
            self.client.search(index=self.index, body=knn_query),
        )

        docs = _rrf_merge(bm25_resp, knn_resp, k=self.k)

        # Find the retriever span (child of the node span)
        retriever_span = None
        node_span = get_current_span()
        if node_span:
            node_span_id = node_span.get_span_context().span_id
            instrumentor = LangChainInstrumentor()
            tracer_obj = getattr(instrumentor, "_tracer", None)
            if tracer_obj:
                spans = getattr(tracer_obj, "_spans_by_run", {})
                for _, s in spans.items():
                    parent = getattr(s, "parent", None)
                    if parent and parent.span_id == node_span_id:
                        retriever_span = s
                        break
        log_embedding_span(docs, parent_span=retriever_span or node_span)

        return docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Not used — async version is preferred."""
        raise NotImplementedError("Use ainvoke() instead")


# --- Clients ---
opensearch_client = AsyncOpenSearch(
    hosts=[{"host": os.environ["HOST"], "port": 443}],
    http_auth=(os.environ["USERNAME"], os.environ["PASSWORD"]),
    use_ssl=True,
    verify_certs=True,
)
retriever = OpenSearchRetriever(client=opensearch_client, index=os.environ["INDEX"])
llm = ChatOpenAI(model="gpt-4o-mini")


# --- Graph state & config ---
class Context(TypedDict):
    """Context parameters for the agent."""

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent."""

    messages: List[AnyMessage] = field(default_factory=list)
    docs: List[Document] = field(default_factory=list)
    search_query: str = ""
    source_file: str = ""


def _extract_query(message: Any) -> str:
    """Extract text content from a message (handles both objects and dicts)."""
    if hasattr(message, "content"):
        return message.content
    if isinstance(message, dict):
        return message.get("content", str(message))
    return str(message)


# --- Graph nodes ---
def parse_input(state: State) -> Dict[str, Any]:
    """Extract question text and optional filename from a multipart message."""
    message = state.messages[-1]
    content = message.content

    # Plain text message — no file attached
    if isinstance(content, str):
        return {
            "messages": [HumanMessage(content=content)],
            "source_file": "",
        }

    # Multipart message — list of {"type": "text", "text": "..."} dicts
    source_file = ""
    text_parts: List[str] = []
    for part in content:
        text = part.get("text", "") if isinstance(part, dict) else str(part)
        if text.lower().endswith(".pdf"):
            source_file = text
        else:
            text_parts.append(text)

    query = " ".join(text_parts).strip()
    return {
        "messages": [HumanMessage(content=query)],
        "source_file": source_file,
    }


async def rewrite_query(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Expand the user question into search-optimized legal terms."""
    query = _extract_query(state.messages[-1])
    chain = query_expansion_prompt | llm | StrOutputParser()
    search_query = await chain.ainvoke({"query": query})
    return {"search_query": search_query.strip()}


async def retrieve(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Retrieve relevant documents from OpenSearch."""
    query = _extract_query(state.messages[-1])
    docs = await retriever.ainvoke(query, search_query=state.search_query)
    return {"docs": docs}


async def rerank(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Rerank retrieved documents using Cohere rerank API."""
    if not state.docs:
        return {"docs": []}
    query = _extract_query(state.messages[-1])
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cohere.com/v2/rerank",
            headers={
                "Authorization": f'Bearer {os.environ["COHERE_API_KEY"]}',
                "Content-Type": "application/json",
            },
            json={
                "model": "rerank-v3.5",
                "query": query,
                "documents": [doc.page_content for doc in state.docs],
                "top_n": len(state.docs),
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
    reranked = []
    for result in data["results"]:
        doc = state.docs[result["index"]]
        doc.metadata["rerank_score"] = result["relevance_score"]
        reranked.append(doc)
    return {"docs": reranked}


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate a response using retrieved context and GPT-4o-mini."""
    query = _extract_query(state.messages[-1])

    context = "\n\n".join(doc.page_content for doc in state.docs)
    chain = rag_prompt | llm | StrOutputParser()
    response = await chain.ainvoke({"query": query, "context": context})

    return {"messages": [AIMessage(content=response)]}


# --- Define the graph ---
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(parse_input)
    .add_node(rewrite_query)
    .add_node(retrieve)
    .add_node(rerank)
    .add_node(call_model)
    .add_edge("__start__", "parse_input")
    .add_edge("parse_input", "rewrite_query")
    .add_edge("rewrite_query", "retrieve")
    .add_edge("retrieve", "rerank")
    .add_edge("rerank", "call_model")
    .compile(name="New Graph")
)
