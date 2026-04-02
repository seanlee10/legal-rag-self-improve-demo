"""Tests for source_file filtering in retrieve node."""

from agent.graph import build_bm25_query, build_knn_query


def test_bm25_query_with_source_filter():
    """BM25 query includes metadata.source term filter when source_file is set."""
    query = build_bm25_query("compliance requirements", fetch_size=30, source_file="report.pdf")
    assert query["query"]["bool"]["filter"] == [{"term": {"metadata.source": "report.pdf"}}]
    assert query["query"]["bool"]["must"][0]["multi_match"]["query"] == "compliance requirements"
    assert query["size"] == 30


def test_bm25_query_without_source_filter():
    """BM25 query has no filter when source_file is empty."""
    query = build_bm25_query("compliance requirements", fetch_size=30, source_file="")
    assert query["query"]["multi_match"]["query"] == "compliance requirements"
    assert "bool" not in query["query"]


def test_knn_query_with_source_filter():
    """kNN query includes metadata.source term filter when source_file is set."""
    embedding = [0.1] * 1024
    query = build_knn_query(embedding, fetch_size=30, source_file="report.pdf")
    assert query["query"]["knn"]["embedding"]["filter"] == {"term": {"metadata.source": "report.pdf"}}


def test_knn_query_without_source_filter():
    """kNN query has no filter when source_file is empty."""
    embedding = [0.1] * 1024
    query = build_knn_query(embedding, fetch_size=30, source_file="")
    assert "filter" not in query["query"]["knn"]["embedding"]
