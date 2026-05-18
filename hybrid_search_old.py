"""Run BM25, kNN, or hybrid retrieval against the news_articles index.

Used for Steps 6 through 9 of the ES hybrid retrieval lab. Supports three
retrieval modes — bm25, vector, and hybrid — with an optional category filter.

Run from inside the project directory with the venv activated:
    python hybrid_search.py --mode bm25 "managing personal finance"
    python hybrid_search.py --mode vector "managing personal finance"
    python hybrid_search.py --mode hybrid "managing personal finance"
    python hybrid_search.py --mode hybrid "managing personal finance" --category MONEY
"""
import argparse
from typing import Iterable

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "news_articles"
ES_HOST = "http://localhost:9200"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# RRF fusion constant — the standard value from the RRF paper.
# Larger values flatten the contribution of higher ranks; smaller values
# emphasise the top ranks more sharply. 60 is the canonical choice.
RRF_K = 60


def run_bm25_search(es: Elasticsearch, query: str, category: str | None, size: int) -> list[dict]:
    """Run a multi_match BM25 query, optionally with a category filter.

    Returns the list of hit dicts from the ES response.
    """
    must_clause = [{
        "multi_match": {
            "query": query,
            "fields": ["headline", "short_description"],
        }
    }]
    filter_clause = []
    if category is not None:
        filter_clause.append({"term": {"category": category}})

    response = es.search(
        index=INDEX_NAME,
        query={"bool": {"must": must_clause, "filter": filter_clause}},
        size=size,
        source_excludes=["embedding"],
    )
    return response["hits"]["hits"]


def run_vector_search(
    es: Elasticsearch, query_vector: list[float], category: str | None, size: int
) -> list[dict]:
    """Run a kNN vector search, optionally with a category filter.

    Returns the list of hit dicts from the ES response.
    """
    knn_clause = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": size,
        "num_candidates": size * 10,
    }
    if category is not None:
        knn_clause["filter"] = {"term": {"category": category}}

    response = es.search(
        index=INDEX_NAME,
        knn=knn_clause,
        size=size,
        source_excludes=["embedding"],
    )
    return response["hits"]["hits"]


def reciprocal_rank_fusion(
    result_lists: Iterable[list[dict]],
    k: int = RRF_K,
    final_size: int = 5,
) -> list[dict]:
    """Combine multiple ranked result lists into a single fused ranking.

    For each document appearing in any input list, compute a fused score
    by summing 1/(k + rank) across all lists where the document appears.
    Documents are identified by their ES _id, and the original hit metadata
    is preserved from whichever list first contained the document.

    The result is a list of hits sorted by fused score descending, truncated
    to final_size.
    """
    fused_scores: dict[str, float] = {}
    seen_hits: dict[str, dict] = {}

    for result_list in result_lists:
        # rank is 1-indexed; the top result in each list has rank 1.
        for rank, hit in enumerate(result_list, start=1):
            doc_id = hit["_id"]
            # RRF formula: contribution to fused score from this list.
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            # Keep the first hit dict we see for each document — we only need
            # one copy of its metadata for the final response.
            if doc_id not in seen_hits:
                seen_hits[doc_id] = hit

    # Sort document IDs by fused score descending and take the top N.
    sorted_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:final_size]

    # Build the final result list, attaching the fused score so it is visible.
    fused_results = []
    for doc_id in sorted_doc_ids:
        hit = seen_hits[doc_id].copy()
        hit["_score"] = fused_scores[doc_id]
        fused_results.append(hit)
    return fused_results


def print_results(hits: list[dict]) -> None:
    """Pretty-print a list of hits in a consistent format across all modes."""
    for i, hit in enumerate(hits, start=1):
        src = hit["_source"]
        print(f"[{i}] score={hit['_score']:.4f}  category={src['category']}  date={src['date']}")
        print(f"    headline: {src['headline']}")
        desc = src['short_description'][:120] + ('...' if len(src['short_description']) > 120 else '')
        print(f"    description: {desc}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval against news_articles.")
    parser.add_argument("query", help="The query string.")
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid).",
    )
    parser.add_argument("--category", default=None, help="Optional category filter.")
    parser.add_argument("--size", type=int, default=5, help="Number of final results.")
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST)

    # For hybrid and vector modes, we need a query embedding.
    # We compute it once even if only one mode needs it — cleaner code.
    query_vector = None
    if args.mode in ("vector", "hybrid"):
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        query_vector = model.encode(args.query, convert_to_numpy=True).tolist()

    print(f"\nMode: {args.mode}  Query: {args.query!r}  Category: {args.category}\n")

    if args.mode == "bm25":
        hits = run_bm25_search(es, args.query, args.category, args.size)
    elif args.mode == "vector":
        hits = run_vector_search(es, query_vector, args.category, args.size)
    else:  # hybrid
        # For hybrid, we fetch more candidates from each method than we will return,
        # so RRF has enough material to do meaningful fusion. Then truncate to size.
        rank_window = max(args.size * 10, 50)
        bm25_hits = run_bm25_search(es, args.query, args.category, rank_window)
        vector_hits = run_vector_search(es, query_vector, args.category, rank_window)
        hits = reciprocal_rank_fusion([bm25_hits, vector_hits], final_size=args.size)

    print_results(hits)


if __name__ == "__main__":
    main()
