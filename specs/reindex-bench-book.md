# Spec: Re-index Bench Book for Complete Passage Coverage

## Problem

The legal-rag-qa evaluation dataset references 95 unique passages from the Victorian bench book. Of these, **38 passages (40%) are missing from the OpenSearch index entirely**, causing a hard ceiling on retrieval performance. The current Recall@10 is 49% — it cannot exceed 60% without fixing the index.

The missing passages are not random. In most cases the *section* exists in the index but specific chunks within it were dropped during ingestion. For example, section `2.5` has chunk `c1-s2` indexed but `c1-s1` is missing; section `3.6` has `c3-s2/s3/s7` but `c1-s1` and `c2-s1` are absent.

## Root Cause

The ingestion pipeline is dropping chunks. Likely causes (investigate in order):

1. **Chunking produces fragments below a size threshold** that get silently filtered out
2. **Duplicate detection** is too aggressive and merges distinct sub-chunks (e.g. `c1-s1` and `c1-s2` within the same section)
3. **Pagination or file reading** truncates sections, so later chunks in a section are never created

## Index Schema

The current OpenSearch index uses the following document structure:

```json
{
  "chunk": "<passage text>",
  "embedding": [<1024-dim float vector>],
  "metadata": {
    "source": "1.2-c2-s2",
    "title": "1.2 Excusing Jurors",
    "footnotes": "[^2]: Ibid."
  }
}
```

- **source ID format**: `<section>-c<chunk>-s<sub>` (e.g. `7.3.2-c5-s1`)
- **embedding model**: `text-embedding-3-large` at 1024 dimensions
- **index currently contains**: 713+ unique source IDs

## Requirements

### 1. Audit the ingestion pipeline

- Trace the full path from raw bench book source files → chunking → embedding → OpenSearch bulk insert
- Log the source ID of every chunk produced and compare against the expected set below
- Identify exactly where the 38 missing chunks are dropped

### 2. Fix the gap

Ensure all chunks from the bench book are indexed. After re-ingestion:

- Every source ID in the "Missing Passages" list below must be present in the index
- No existing passages should be removed (additive fix)
- The same metadata schema (`source`, `title`, `footnotes`) must be preserved
- Embeddings must use `text-embedding-3-large` at 1024 dimensions to match existing vectors

### 3. Validate

After re-indexing, run this verification query for each missing passage:

```bash
# Verify a passage exists
curl -XPOST "$OPENSEARCH_HOST/$INDEX/_search" -H 'Content-Type: application/json' -d '{
  "query": { "term": { "metadata.source.keyword": "SOURCE_ID" } },
  "size": 1
}'
```

All 38 passages should return exactly 1 hit.

## Missing Passages

The following 38 source IDs are expected by the evaluation dataset but are not present in the OpenSearch index:

```
1.5-c6-s1
2.1-c1-s1
2.3.3-c2-s1
2.5-c1-s1
3.6-c1-s1
3.6-c2-s1
4.2-c3-s1
4.6-c3-s5
4.6-c4-s1
4.11-c8-s1
4.12.2-c1-s2
4.13.2-c4-s2
4.16-c1-s1
4.18-c2-s1
4.23-c2-s1
5.1-c2-s1
5.9-c1-s2
6.2-c9-s1
7.1.1-c3-s1
7.1.1-c6-s2
7.2.1-c2-s1
7.2.1A-c3-s1
7.2.5-c3-s8
7.2.5-c8-s1
7.2.6-c5-s5
7.3.1.2-c4-s10
7.3.1.2-c8-s6
7.3.1.7-c1-s1
7.3.2-c3-s1
7.3.2-c5-s1
7.3.13.5-c4-s6
7.4.9-c5-s2
7.4.12-c2-s5
7.4.17-c4-s1
7.5.10-c2-s1
7.5.15-c3-s1
7.5.15-c4-s2
7.5.18-c2-s1
```

### Near-matches in index (evidence of partial ingestion)

These show the gap is within sections that are partially indexed:

| Missing Passage | Existing in Index (same section) |
|---|---|
| `1.5-c6-s1` | `1.5-c1-s1`, `c1-s2`, `c1-s3`, `c1-s4`, `c5-s1`, `c7-s2`, `c8-s1` |
| `2.1-c1-s1` | `2.1-c3-s1`, `c6-s1` |
| `2.3.3-c2-s1` | `2.3.3-c2-s2` |
| `2.5-c1-s1` | `2.5-c1-s2` |
| `3.6-c1-s1`, `c2-s1` | `3.6-c3-s2`, `c3-s3`, `c3-s7` |
| `4.11-c8-s1` | `4.11-c6-s1`, `c7-s1`, `c9-s2`, `c9-s5`, `c9-s6` ... |
| `4.12.2-c1-s2` | `4.12.2-c1-s1` |
| `4.13.2-c4-s2` | `4.13.2-c1-s1`, `c2-s1`, `c2-s2`, `c2-s4` |

## Expected Impact

- Current Recall@10: **49/100**
- Theoretical max after fix: **95/100** (some may still rank outside top 10)
- Current eval pass rate: **19/100** (limited by Recall@1)
- Expected improvement: significant — the 38 missing passages account for 38 of the 51 retrieval failures

## Out of Scope

- Retriever code changes (already improved with RRF + dedup in this repo)
- Chunking strategy changes beyond fixing the dropped chunks
- Embedding model changes
