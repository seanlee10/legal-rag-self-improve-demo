"""Tests for summarization node helpers."""

from langchain_core.documents import Document

from agent.graph import _chunk_documents, _estimate_tokens, SUMMARIZE_TOKEN_THRESHOLD


def test_estimate_tokens():
    """Token estimation returns roughly 1 token per 4 characters."""
    assert _estimate_tokens("hello world") == 2  # 11 chars / 4 = 2


def test_estimate_tokens_empty():
    """Empty string returns 0 tokens."""
    assert _estimate_tokens("") == 0


def test_chunk_documents_short():
    """Documents under threshold return a single group."""
    docs = [Document(page_content="short text")]
    groups = _chunk_documents(docs, max_tokens=1000)
    assert len(groups) == 1
    assert groups[0] == docs


def test_chunk_documents_long():
    """Documents over threshold are split into multiple groups."""
    # Create docs that exceed max_tokens when combined
    docs = [Document(page_content="x" * 400) for _ in range(10)]
    # Each doc is ~100 tokens, 10 docs = ~1000 tokens; threshold at 500
    groups = _chunk_documents(docs, max_tokens=500)
    assert len(groups) > 1
    # All docs accounted for
    flat = [doc for group in groups for doc in group]
    assert len(flat) == 10


def test_summarize_token_threshold_is_10000():
    """Threshold is set to 10000 tokens (~15 pages)."""
    assert SUMMARIZE_TOKEN_THRESHOLD == 10000
