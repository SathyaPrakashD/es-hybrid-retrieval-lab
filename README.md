# es-hybrid-retrieval-lab

A measured comparison of BM25, dense vector search, and hybrid retrieval
(weighted Reciprocal Rank Fusion) in Elasticsearch 9.4, run over 50,000 news
articles from the HuffPost News Category Dataset.

**Finding:** On a 15-query stratified eval of paraphrastic queries, dense vector
search outperformed hybrid retrieval (87% vs 67% recall@5). RRF fusion dragged
the better retriever down rather than lifting both up, because BM25's underlying
recall (33%) was too weak to contribute useful rank signal.

Full discussion of methodology and caveats below.

---

## Table of Contents

- [What you'll learn](#what-youll-learn)
- [Background concepts](#background-concepts)
- [Headline numbers](#headline-numbers)
- [What this project is](#what-this-project-is)
- [What the numbers mean](#what-the-numbers-mean)
- [Methodology and caveats](#methodology-and-caveats)
- [How to reproduce](#how-to-reproduce)
- [Repository structure](#repository-structure)
- [Architectural choices worth flagging](#architectural-choices-worth-flagging)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## What you'll learn

By working through this lab you will understand:

- The difference between **BM25** (keyword-based) and **vector search** (semantic/embedding-based) retrieval, and when each excels.
- How **Reciprocal Rank Fusion (RRF)** combines multiple ranked result lists into a single fused ranking.
- Why hybrid retrieval is not always better than its best individual component — and how to measure the difference.
- How to run Elasticsearch locally with Docker and index documents with precomputed embeddings.
- How to design a simple recall-based evaluation and interpret the results honestly.

This project is intentionally scoped to run on a laptop over a weekend. No cloud account, no GPU, and no paid Elasticsearch tier are required.

---

## Background concepts

If you are new to search or information retrieval, here is a brief primer on the three retrieval methods compared in this lab.

**BM25** is the classical keyword search algorithm used by Elasticsearch and many search engines. It ranks documents by how frequently the query terms appear in a document, weighted by how rare those terms are across the whole corpus. BM25 works well when the query and document share the same vocabulary, but struggles with paraphrastic queries — cases where the meaning is the same but different words are used.

**Dense vector search (semantic search)** converts both documents and queries into numerical vectors using a machine learning embedding model. Documents that are semantically similar end up close together in vector space, regardless of the exact words used. This is the strength that BM25 lacks.

**Reciprocal Rank Fusion (RRF)** is a simple algorithm for combining ranked lists from different retrieval methods. For each document, it computes a fused score by summing `1 / (k + rank)` across all lists where the document appears, where `k` is a smoothing constant (default 60). The intuition: a document that ranks highly in multiple lists should rank highly in the final merged list. Weighted RRF lets you assign different trust levels to each method.

---

## Headline numbers

Recall@5 across six retrieval configurations, n=15 queries:

| Retrieval mode               | Recall@5      |
|------------------------------|---------------|
| BM25 (no filter)             | 33.3% (5/15)  |
| BM25 (category filter)       | 33.3% (5/15)  |
| Vector (no filter)           | 86.7% (13/15) |
| Vector (category filter)     | 93.3% (14/15) |
| Hybrid RRF (no filter)       | 66.7% (10/15) |
| Hybrid RRF (category filter) | 66.7% (10/15) |

Stratified by category (n=3 queries per category):

| Mode              | POLITICS | ENTERTAIN | BUSINESS | TRAVEL | MONEY |
|-------------------|----------|-----------|----------|--------|-------|
| BM25 unfiltered   | 1/3      | 0/3       | 2/3      | 0/3    | 2/3   |
| BM25 filtered     | 1/3      | 0/3       | 2/3      | 0/3    | 2/3   |
| Vector unfiltered | 3/3      | 2/3       | 2/3      | 3/3    | 3/3   |
| Vector filtered   | 3/3      | 2/3       | 3/3      | 3/3    | 3/3   |
| Hybrid unfiltered | 2/3      | 2/3       | 2/3      | 1/3    | 3/3   |
| Hybrid filtered   | 2/3      | 2/3       | 2/3      | 1/3    | 3/3   |

---

## What this project is

A lab exercise to measure — rather than assume — the value of common retrieval architecture choices: pre-filtering, hybrid retrieval, weighted RRF fusion. Built to make a specific architectural question answerable with real data on a laptop in one weekend.

The retrieval pipeline:

- **Corpus:** 50,000 news articles from the [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset), spanning 41 categories from POLITICS (17,399 articles) down to WEDDINGS (2).
- **Index:** Elasticsearch 9.4 with explicit mapping — `keyword` for filterable fields, `text` for BM25-searchable fields, `dense_vector` (384-dim, cosine similarity) for semantic search.
- **Embeddings:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), computed at index time over a concatenation of `headline` and `short_description`.
- **Retrieval:** Three modes — BM25 alone, kNN vector search alone, and weighted RRF hybrid (BM25 weight 0.5, vector weight 1.0, fusion constant k=60).
- **Filtering:** Optional pre-filter on the `category` field for any mode.
- **Eval:** 15 hand-crafted query-relevance pairs stratified across 5 categories of varying size (POLITICS, ENTERTAINMENT, BUSINESS, TRAVEL, MONEY).

---

## What the numbers mean

**Dense vector retrieval essentially solves paraphrastic news search at this scale.** BGE-small with no filter hits 87% recall@5; with a category filter it hits 93%. For workloads where queries are natural-language paraphrases of document content, a modern embedding model is doing nearly all the work.

**Hybrid retrieval underperformed vector alone.** This is the most architecturally interesting finding. RRF is a fusion algorithm — it cannot detect that one of its inputs is unreliable. With BM25 at 33% recall, its top-ranked false positives were getting fused with vector search's top-ranked true positives, and the fusion math has no concept of "trust this method less when it's wrong." Some of vector search's correct top-5 results were pushed below rank 5 in the fused output by high-confidence BM25 noise.

**Pre-filtering contributed almost nothing.** The eval was designed expecting the category filter to be the architectural payoff, particularly for small categories like MONEY (49 articles total) where the filter eliminates 99.9% of the candidate pool. In practice, the filter added one query's worth of improvement on vector search (87% → 93%) and zero on every other mode. When your retrieval is already topically coherent — as vector search consistently was — filtering has no room to help.

---

## Methodology and caveats

**The eval was deliberately biased toward paraphrastic retrieval.** Queries in the golden set were hand-constructed to share minimal lexical tokens with their target documents — for example, the query *"how to save money for irregular expenses throughout the year"* targets an article titled *"What Is A Sinking Fund — And Why Should You Have One?"*. This is the exact failure mode for BM25 (vocabulary mismatch) and the exact strength of vector search (semantic similarity across different vocabulary). On a workload with more lexical-overlap queries, BM25 would perform substantially better and hybrid retrieval would likely earn its complexity.

**The conclusion is "vector search dominates this workload," not "vector search dominates universally."** Production retrieval systems typically see a mixed query distribution — some highly paraphrastic, some heavily lexical — and the right architecture is workload-dependent. The point of this eval is that measurement should drive the architecture choice, not the other way around.

**Sample size is small.** With 15 queries, a single query going from miss to hit moves recall@5 by 6.7 percentage points. The large effects in the table (BM25 vs vector, ~50pp gap) are robust at this sample size. The small effects (the 6.7pp filter improvement on vector search, the gap between hybrid filtered vs unfiltered) are within sampling noise and should not be over-interpreted.

**Relevance judgments are single-annotator.** I selected the "correct" document for each query based on my own judgment of paraphrastic relevance. A second annotator might disagree on individual calls. The full golden set is in `golden_set.json` for anyone who wants to inspect or dispute specific pairs.

---

## How to reproduce

### Prerequisites

- [Docker Desktop 4.37+](https://www.docker.com/products/docker-desktop/) (for Elasticsearch + Kibana)
- Python 3.10+
- ~2 GB free disk for the Docker images, ~200 MB for the embedding model cache, ~150 MB for the index data
- A [Kaggle account](https://www.kaggle.com/) to download the News Category Dataset

> **New to Docker?** Docker lets you run software like Elasticsearch in an isolated container without installing it directly on your machine. After installing Docker Desktop, make sure it is running (the Docker icon should be visible in your system tray) before proceeding.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/SathyaPrakashD/es-hybrid-retrieval-lab.git
cd es-hybrid-retrieval-lab

# 2. Start Elasticsearch 9.4.1 and Kibana via docker-compose
docker compose up -d

# 3. Confirm ES is up (should return JSON with "status": "green" or "yellow")
curl 'http://localhost:9200/_cluster/health?pretty'

# 4. Set up Python environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download the dataset

Download the [News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle. Unzip the file and save it as `News_Category_Dataset_v3.json` in the repo root.

> **Why isn't the dataset in the repo?** The file is ~83 MB, which is too large to store in git efficiently. The `.gitignore` excludes it. You must download it manually from Kaggle (free account required).

### Index the corpus

```bash
# Computes 384-dim BGE-small embeddings for 50K articles and bulk-indexes
# into the news_articles index. Takes ~15-25 minutes on CPU.
python index_data.py
```

This step downloads the embedding model (~130 MB) on first run and caches it locally. Subsequent runs use the cache.

### Run the eval

```bash
# Loads golden_set.json, runs 6 retrieval modes per query, prints the
# overall and stratified recall@5 tables.
python run_eval.py
```

### Explore interactively

```bash
# Run a single query in any retrieval mode, with optional category filter
# and weighted RRF.
python hybrid_search.py --mode bm25 "managing personal finance"
python hybrid_search.py --mode vector "managing personal finance"
python hybrid_search.py --mode hybrid "managing personal finance" \
  --bm25-weight 0.5 --vector-weight 1.0 --category MONEY
```

You can also open Kibana at <http://localhost:5601> to browse the index and run queries interactively through a web UI.

---

## Repository structure

```
.
├── docker-compose.yml      # ES 9.4.1 + Kibana, single-node, security off
├── mapping.json            # ES index mapping for news_articles
├── requirements.txt        # Python dependencies (pinned)
├── index_data.py           # Load + embed + bulk-index the Kaggle dataset
├── hybrid_search.py        # CLI: BM25 / vector / weighted-RRF hybrid retrieval
├── vector_search.py        # Standalone vector-only search script
├── browse_category.py      # Helper: sample random articles from a category
├── embed_query.py          # Helper: emit a 384-dim query vector for Kibana Dev Tools
├── run_eval.py             # Eval runner: 6 modes x 15 queries -> recall@5
├── golden_set.json         # 15 hand-labelled query-relevance pairs
└── README.md               # This file
```

> **Note:** `hybrid_search_old.py` is a historical snapshot from before weighted RRF was added. It implements standard (unweighted) RRF and is kept for reference. Use `hybrid_search.py` for all new work.

---

## Architectural choices worth flagging

A few decisions in the implementation that are worth knowing about, in case anyone wants to fork or extend the work.

**Native RRF requires a platinum license.** Elasticsearch 8.9+ ships a built-in `retriever` clause with native RRF fusion, but it's restricted to the platinum subscription tier — the basic tier returns a `403` with `license.expired.feature`. The fusion in this repo is implemented in Python in `hybrid_search.py` and explicitly weights the two sub-retrievers, which lets it run on basic ES and makes the fusion math inspectable rather than opaque.

**Embeddings are computed application-side, not server-side.** The `dense_vector` field stores precomputed embeddings; ES does not run the embedding model. Query embeddings are computed in Python before being sent to ES as a JSON array. Production deployments would typically host the embedding model in the cluster via the ML inference node, but that requires a paid tier.

**No reranker, no LLM, no agent.** The pipeline stops at retrieval and is deliberately not extended into downstream synthesis. The architectural question being measured is "does fusion + filtering improve retrieval recall," and adding a reranker or LLM would conflate that question with separate ones.

---

## Troubleshooting

**`docker compose up` fails or ES does not start**
Make sure Docker Desktop is running. On first start, ES can take 60–90 seconds to become healthy. Run `docker compose logs elasticsearch` to see startup messages.

**`curl http://localhost:9200` returns connection refused**
Wait for ES to finish initializing, then retry. Check `docker ps` to confirm the container is running.

**`python index_data.py` fails with `Cannot reach Elasticsearch`**
ES is not yet up or the port is not mapped correctly. Run `curl 'http://localhost:9200/_cluster/health?pretty'` and confirm you get a JSON response before running the indexer.

**Embedding step is very slow**
15–25 minutes on CPU is expected for 50,000 documents. The model runs each batch of 64 texts through a neural network forward pass. You can lower `EMBED_BATCH_SIZE` in `index_data.py` if you run out of memory, or raise it (up to ~256) if you have a GPU.

**`run_eval.py` shows 0 hits for every query**
The index may be empty. Check the document count with:
```bash
curl 'http://localhost:9200/news_articles/_count?pretty'
```
If the count is 0, re-run `index_data.py`. If the index does not exist at all, make sure you ran `index_data.py` to completion first.

**Port 9200 or 5601 is already in use**
Another service is using the port. Stop the conflicting service, or edit `docker-compose.yml` to map to different host ports (e.g., `"9201:9200"`), then update `ES_HOST` in the Python scripts accordingly.

---

## Acknowledgements

- **News Category Dataset:** Rishabh Misra, https://www.kaggle.com/datasets/rmisra/news-category-dataset
- **Embedding model:** BAAI/bge-small-en-v1.5, https://huggingface.co/BAAI/bge-small-en-v1.5
- **Reciprocal Rank Fusion:** Cormack et al., 2009. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

---

## License

MIT. See [LICENSE](LICENSE) file.
