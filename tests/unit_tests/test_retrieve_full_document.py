"""Tests for full-document retrieval query building."""

from agent.graph import build_full_document_query


def test_full_document_query_structure():
    """Query fetches all chunks matching a source file, sorted by _id."""
    query = build_full_document_query("report.pdf")
    assert query["query"]["term"]["metadata.source"] == "report.pdf"
    assert query["size"] == 10000
    assert query["sort"] == [{"_id": "asc"}]


def test_full_document_query_different_file():
    """Query uses the provided filename."""
    query = build_full_document_query("circular-2024.pdf")
    assert query["query"]["term"]["metadata.source"] == "circular-2024.pdf"
