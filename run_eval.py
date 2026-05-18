"""Run the recall@5 evaluation over the golden set across six retrieval modes.

Loads golden_set.json, runs each query through BM25, vector, and hybrid retrieval,
both with and without a category filter, and prints a recall@5 summary table.

Run from inside the project directory with the venv activated:
    python run_eval.py
"""
import json
from collections import defaultdict
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Import the retrieval functions from our existing search script.
# This keeps the eval logic separate from the search implementation and
# guarantees the eval runs the same code paths the live searches would.
from hybrid_search import (
    run_bm25_search,
    run_vector_search,
    weighted_reciprocal_rank_fusion,
    INDEX_NAME,
    ES_HOST,
    EMBED_MODEL_NAME,
)

GOLDEN_SET_FILE = Path("golden_set.json")
TOP_K = 5

# Hybrid mode uses vector-heavy weighting based on the manual exploration
# showing this works better for paraphrastic news queries.
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 1.0


def hit_at_k(hits: list[dict], expected_id: str, k: int = TOP_K) -> bool:
    """Return True if the expected document ID appears in the top k hits."""
    top_k_ids = [hit["_id"] for hit in hits[:k]]
    return expected_id in top_k_ids


def evaluate_query(
    es: Elasticsearch,
    model: SentenceTransformer,
    query: str,
    expected_id: str,
    category: str | None,
) -> dict[str, bool]:
    """Run all six retrieval modes for one query and check whether the
    expected document was retrieved in the top 5 for each mode.

    Returns a dict keyed by mode name with True/False values.
    """
    # Compute the query embedding once and reuse it across vector and hybrid modes.
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    # Window size used for hybrid mode — pull more candidates than final top-k
    # so RRF has material to fuse meaningfully.
    rank_window = 50

    # Six retrieval modes: three methods (bm25/vector/hybrid) × two filter states (off/on).
    results = {}

    # BM25, no filter.
    bm25_unfiltered = run_bm25_search(es, query, None, TOP_K)
    results["bm25_unfiltered"] = hit_at_k(bm25_unfiltered, expected_id)

    # BM25, filtered to expected category.
    bm25_filtered = run_bm25_search(es, query, category, TOP_K)
    results["bm25_filtered"] = hit_at_k(bm25_filtered, expected_id)

    # Vector search, no filter.
    vector_unfiltered = run_vector_search(es, query_vector, None, TOP_K)
    results["vector_unfiltered"] = hit_at_k(vector_unfiltered, expected_id)

    # Vector search, filtered.
    vector_filtered = run_vector_search(es, query_vector, category, TOP_K)
    results["vector_filtered"] = hit_at_k(vector_filtered, expected_id)

    # Hybrid (weighted RRF), no filter.
    bm25_hits_wide = run_bm25_search(es, query, None, rank_window)
    vector_hits_wide = run_vector_search(es, query_vector, None, rank_window)
    hybrid_unfiltered = weighted_reciprocal_rank_fusion(
        [bm25_hits_wide, vector_hits_wide],
        [BM25_WEIGHT, VECTOR_WEIGHT],
        final_size=TOP_K,
    )
    results["hybrid_unfiltered"] = hit_at_k(hybrid_unfiltered, expected_id)

    # Hybrid (weighted RRF), filtered.
    bm25_hits_filtered_wide = run_bm25_search(es, query, category, rank_window)
    vector_hits_filtered_wide = run_vector_search(es, query_vector, category, rank_window)
    hybrid_filtered = weighted_reciprocal_rank_fusion(
        [bm25_hits_filtered_wide, vector_hits_filtered_wide],
        [BM25_WEIGHT, VECTOR_WEIGHT],
        final_size=TOP_K,
    )
    results["hybrid_filtered"] = hit_at_k(hybrid_filtered, expected_id)

    return results


def main() -> None:
    # Load the golden set from JSON.
    with GOLDEN_SET_FILE.open("r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"Loaded {len(golden_set)} query-relevance pairs from {GOLDEN_SET_FILE}\n")

    # Connect to ES and load the embedding model once.
    es = Elasticsearch(ES_HOST)
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print()

    # Track per-mode hit counts in two ways: overall and stratified by category.
    overall_hits: dict[str, int] = defaultdict(int)
    per_category_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_category_total: dict[str, int] = defaultdict(int)

    # Run the eval over every query in the golden set.
    print(f"Running eval over {len(golden_set)} queries...\n")
    for i, entry in enumerate(golden_set, start=1):
        query = entry["query"]
        expected_id = entry["expected_id"]
        category = entry["expected_category"]

        print(f"[{i}/{len(golden_set)}] {category}: {query[:60]}...")
        results = evaluate_query(es, model, query, expected_id, category)

        per_category_total[category] += 1
        for mode, hit in results.items():
            if hit:
                overall_hits[mode] += 1
                per_category_hits[category][mode] += 1

    # Print the overall recall@5 table.
    total = len(golden_set)
    print("\n" + "=" * 70)
    print(f"OVERALL RECALL@5 (n={total})")
    print("=" * 70)
    print(f"{'Mode':<25} {'Hits':<8} {'Recall@5':<10}")
    print("-" * 70)
    for mode in ["bm25_unfiltered", "bm25_filtered",
                 "vector_unfiltered", "vector_filtered",
                 "hybrid_unfiltered", "hybrid_filtered"]:
        hits = overall_hits[mode]
        recall = hits / total
        print(f"{mode:<25} {hits:<8} {recall:.3f}")

    # Print the stratified table by category.
    print("\n" + "=" * 70)
    print("RECALL@5 BY CATEGORY (stratified)")
    print("=" * 70)
    categories_in_order = ["POLITICS", "ENTERTAINMENT", "BUSINESS", "TRAVEL", "MONEY"]
    header = f"{'Mode':<25}" + "".join(f"{c[:8]:<10}" for c in categories_in_order)
    print(header)
    print("-" * len(header))
    for mode in ["bm25_unfiltered", "bm25_filtered",
                 "vector_unfiltered", "vector_filtered",
                 "hybrid_unfiltered", "hybrid_filtered"]:
        row = f"{mode:<25}"
        for cat in categories_in_order:
            hits = per_category_hits[cat][mode]
            total_cat = per_category_total[cat]
            recall = hits / total_cat if total_cat > 0 else 0.0
            row += f"{recall:.2f} ({hits}/{total_cat})  ".ljust(10)
        print(row)
    print()


if __name__ == "__main__":
    main()
