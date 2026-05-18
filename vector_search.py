"""Embed a query and run a kNN search against the news_articles index.

Used for Step 7 of the ES hybrid retrieval lab. Takes a query string,
embeds it with the same BGE-small model used at index time, then runs
a kNN search and pretty-prints the top results.

Run from inside the project directory with the venv activated:
    python vector_search.py "managing personal finance"
    python vector_search.py "managing personal finance" --category MONEY
"""
import argparse
import json

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Same configuration as the indexing script — critical that the model
# matches what was used at index time, otherwise vectors are in different spaces.
INDEX_NAME = "news_articles"
ES_HOST = "http://localhost:9200"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_knn_query(query_vector: list[float], category: str | None, k: int = 5) -> dict:
    """Construct the kNN search clause, optionally with a category pre-filter.

    The `num_candidates` parameter controls how aggressively HNSW searches —
    higher values give better accuracy at the cost of latency. A common rule
    of thumb is num_candidates = 10x the final k, which gives good recall
    with minimal latency impact on small indices.
    """
    knn_clause = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": k * 10,
    }
    # Pre-filtering: if a category was specified, restrict the kNN search
    # to documents matching that category before computing nearest neighbours.
    # This is the architectural pattern the whole project is about.
    if category is not None:
        knn_clause["filter"] = {
            "term": {"category": category}
        }
    return knn_clause


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector search against news_articles.")
    parser.add_argument("query", help="The query string to embed and search.")
    parser.add_argument(
        "--category",
        help="Optional category to filter on (e.g., MONEY, PARENTS).",
        default=None,
    )
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    args = parser.parse_args()

    # Load the model — first run downloads it (~130MB), subsequent runs use cache.
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Embed the query into a 384-dimensional vector.
    # convert_to_numpy=False gives us a plain Python list which is what ES expects.
    print(f"Embedding query: {args.query!r}")
    query_vector = model.encode(args.query, convert_to_numpy=True).tolist()

    # Connect to Elasticsearch and run the search.
    es = Elasticsearch(ES_HOST)
    knn = build_knn_query(query_vector, args.category, k=args.k)

    print(f"Running kNN search (k={args.k}, category={args.category})...")
    response = es.search(
        index=INDEX_NAME,
        knn=knn,
        # _source filter: exclude the embedding field from results to keep output readable.
        # The embedding is 384 floats per document; we don't need to see it for sanity-checking.
        source_excludes=["embedding"],
    )

    # Pretty-print the top results.
    print(f"\nTook {response['took']}ms, total hits: {response['hits']['total']['value']}\n")
    for i, hit in enumerate(response["hits"]["hits"], start=1):
        src = hit["_source"]
        print(f"[{i}] score={hit['_score']:.4f}  category={src['category']}  date={src['date']}")
        print(f"    headline: {src['headline']}")
        desc = src['short_description'][:120] + ('...' if len(src['short_description']) > 120 else '')
        print(f"    description: {desc}")
        print()


if __name__ == "__main__":
    main()
