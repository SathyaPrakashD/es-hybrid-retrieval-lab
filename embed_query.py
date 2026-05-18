"""Embed a query string and print the resulting vector as a JSON array.

Used when you want to run vector search from Kibana's Dev Tools console
rather than from Python. Copy the printed array into a knn query.

Run from inside the project directory with the venv activated:
    python embed_query.py "managing personal finance"
"""
import argparse
import json

from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a query string for Kibana.")
    parser.add_argument("query", help="The query string to embed.")
    args = parser.parse_args()

    # Load the same model used at index time — non-negotiable.
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Embed and convert to a plain Python list for clean JSON serialisation.
    vector = model.encode(args.query, convert_to_numpy=True).tolist()

    # Print as a compact JSON array (no whitespace between elements) so it
    # is easy to paste into a Kibana query body.
    print(json.dumps(vector))


if __name__ == "__main__":
    main()
