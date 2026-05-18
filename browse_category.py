"""Browse a random sample of articles from a given category.

Used during golden set construction to find topically distinctive articles
that can serve as relevance anchors for hand-crafted test queries.

Run from inside the project directory with the venv activated:
    python browse_category.py POLITICS
    python browse_category.py MONEY --size 20
"""
import argparse

from elasticsearch import Elasticsearch

INDEX_NAME = "news_articles"
ES_HOST = "http://localhost:9200"


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse random articles from a category.")
    parser.add_argument("category", help="The category to browse (e.g., POLITICS).")
    parser.add_argument("--size", type=int, default=10, help="How many articles to show.")
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST)

    # Use function_score with random_score to get a randomised sample.
    # Without random_score, ES would return results in some deterministic
    # internal order, which would always show us the same articles each run.
    response = es.search(
        index=INDEX_NAME,
        query={
            "function_score": {
                "query": {"term": {"category": args.category}},
                "random_score": {},
            }
        },
        size=args.size,
        source_excludes=["embedding"],
    )

    print(f"\nSample of {args.size} random articles from category: {args.category}\n")
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        print(f"_id: {hit['_id']}")
        print(f"    headline: {src['headline']}")
        desc = src['short_description'][:150] + ('...' if len(src['short_description']) > 150 else '')
        print(f"    description: {desc}")
        print(f"    date: {src['date']}")
        print()


if __name__ == "__main__":
    main()
