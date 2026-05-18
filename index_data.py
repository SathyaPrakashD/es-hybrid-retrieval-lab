"""Load the News Category dataset into Elasticsearch with embeddings.

Reads the raw JSON line by line, computes 384-dim BGE-small embeddings
in batches, and bulk-indexes into the news_articles index.

Run from inside the project directory with the venv activated:
    python index_data.py
"""
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────
# All the tweakable knobs live here so you can adjust without hunting through code.
DATA_FILE = Path("News_Category_Dataset_v3.json")
INDEX_NAME = "news_articles"
ES_HOST = "http://localhost:9200"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_DOCS = 50_000       # Subsample size; set to None to index the full ~210K
EMBED_BATCH_SIZE = 64   # How many docs to embed per model.encode() call
BULK_BATCH_SIZE = 500   # How many docs to send per ES bulk request


def load_documents(path: Path, limit: int | None = None) -> list[dict]:
    """Read the JSON file line by line and keep only the fields our mapping expects.

    The Kaggle file is JSONL (one JSON object per line), not a single JSON array,
    so we parse line-by-line rather than json.load()-ing the whole file.
    """
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            raw = json.loads(line)
            # Keep only the fields declared in our mapping; drop link & authors.
            docs.append({
                "headline": raw["headline"],
                "short_description": raw["short_description"],
                "category": raw["category"],
                "date": raw["date"],
            })
    return docs


def compute_embeddings(docs: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """Embed the headline + short_description for each document.

    We concatenate the two text fields with a separator because the article's
    headline alone often lacks enough signal for semantic similarity, while the
    short_description alone misses the headline's framing. Concatenation gives
    the embedding model both signals in one pass.
    """
    texts = [
        f"{d['headline']}. {d['short_description']}"
        for d in docs
    ]
    # show_progress_bar=True gives a nice progress display during the slow part.
    # batch_size lets the model process multiple texts per forward pass for speed.
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # Convert numpy arrays to plain Python lists because ES expects JSON-serialisable values.
    return embeddings.tolist()


def make_bulk_actions(docs: list[dict], embeddings: list[list[float]]):
    """Generator that yields one ES bulk action per document.

    The `bulk` helper expects a stream of dicts where each dict describes
    one indexing operation. Using a generator means we don't have to build
    the full list in memory all at once — important if you ever scale this
    to millions of documents.
    """
    for doc, emb in zip(docs, embeddings):
        yield {
            "_index": INDEX_NAME,
            "_source": {**doc, "embedding": emb},
        }


def main() -> None:
    # ─── Step 1: connect to ES and verify it's reachable ──────────────────
    print(f"Connecting to Elasticsearch at {ES_HOST}")
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise RuntimeError("Cannot reach Elasticsearch. Is `docker compose up` running?")

    # ─── Step 2: load the raw documents ───────────────────────────────────
    print(f"Loading up to {MAX_DOCS:,} documents from {DATA_FILE}")
    docs = load_documents(DATA_FILE, limit=MAX_DOCS)
    print(f"Loaded {len(docs):,} documents.")

    # ─── Step 3: load the embedding model (downloads ~130MB on first run) ──
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # ─── Step 4: compute embeddings (this is the slow part — 15-25 min on CPU) ──
    print(f"Computing embeddings for {len(docs):,} documents...")
    embeddings = compute_embeddings(docs, model)
    print(f"Done. Embedding dim: {len(embeddings[0])}")

    # ─── Step 5: bulk-index to Elasticsearch ──────────────────────────────
    print(f"Bulk-indexing into {INDEX_NAME} (batch size {BULK_BATCH_SIZE})...")
    success, failures = bulk(
        es,
        make_bulk_actions(docs, embeddings),
        chunk_size=BULK_BATCH_SIZE,
        request_timeout=60,
    )
    print(f"Indexed {success:,} documents. Failures: {len(failures) if isinstance(failures, list) else failures}")

    # ─── Step 6: verify the count ─────────────────────────────────────────
    # ES indexing is asynchronous; force a refresh so the count is accurate immediately.
    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)["count"]
    print(f"Final document count in {INDEX_NAME}: {count:,}")


if __name__ == "__main__":
    main()
