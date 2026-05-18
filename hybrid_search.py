"""Run BM25, kNN, or hybrid retrieval against the news_articles index.

Supports three retrieval modes — bm25, vector, and hybrid — with an optional
category filter. Hybrid mode uses Weighted Reciprocal Rank Fusion, where each
retrieval method can be assigned a different trust weight.

Run from inside the project directory with the venv activated:
    python hybrid_search.py --mode bm25 "managing personal finance"
    python hybrid_search.py --mode vector "managing personal finance"
    python hybrid_search.py --mode hybrid "managing personal finance"
    python hybrid_search.py --mode hybrid "managing personal finance" \\
        --bm25-weight 0.5 --vector-weight 1.0
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
    """Run a multi_match BM25 query, optionally with a category filter."""
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
    """Run a kNN vector search, optionally with a category filter."""
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


def weighted_reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    weights: list[float],
    k: int = RRF_K,
    final_size: int = 5,
) -> list[dict]:
    """Combine multiple ranked result lists into a single fused ranking,
    with per-method weights controlling how much each list contributes.

    For each document appearing in any input list, compute a fused score
    by summing weight_i * 1/(k + rank_i) across all lists where the
    document appears. Documents are identified by their ES _id.

    Setting all weights to 1.0 recovers standard (unweighted) RRF.
    Higher weight means the method's rankings count for more in the fusion.
    """
    if len(result_lists) != len(weights):
        raise ValueError("Number of result lists must match number of weights.")

    fused_scores: dict[str, float] = {}
    seen_hits: dict[str, dict] = {}

    for result_list, weight in zip(result_lists, weights):
        for rank, hit in enumerate(result_list, start=1):
            doc_id = hit["_id"]
            # The weighted RRF formula: scale this list's contribution by its weight.
            contribution = weight * (1.0 / (k + rank))
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + contribution
            if doc_id not in seen_hits:
                seen_hits[doc_id] = hit

    sorted_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:final_size]

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
    # Weighted RRF parameters — only used in hybrid mode.
    # Defaults of 1.0 each recover the standard unweighted RRF behaviour.
    parser.add_argument(
        "--bm25-weight", type=float, default=1.0,
        help="Weight for the BM25 sub-retriever in hybrid fusion (default: 1.0).",
    )
    parser.add_argument(
        "--vector-weight", type=float, default=1.0,
        help="Weight for the vector sub-retriever in hybrid fusion (default: 1.0).",
    )
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST)

    query_vector = None
    if args.mode in ("vector", "hybrid"):
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        model = SentenceTransformer(EMBED_MODEL_NAME)
        query_vector = model.encode(args.query, convert_to_numpy=True).tolist()

    print(f"\nMode: {args.mode}  Query: {args.query!r}  Category: {args.category}")
    if args.mode == "hybrid":
        print(f"Weights: bm25={args.bm25_weight}, vector={args.vector_weight}")
    print()

    if args.mode == "bm25":
        hits = run_bm25_search(es, args.query, args.category, args.size)
    elif args.mode == "vector":
        hits = run_vector_search(es, query_vector, args.category, args.size)
    else:
        rank_window = max(args.size * 10, 50)
        bm25_hits = run_bm25_search(es, args.query, args.category, rank_window)
        vector_hits = run_vector_search(es, query_vector, args.category, rank_window)
        hits = weighted_reciprocal_rank_fusion(
            [bm25_hits, vector_hits],
            [args.bm25_weight, args.vector_weight],
            final_size=args.size,
        )

    print_results(hits)


if __name__ == "__main__":
    main()
