---
title: "legal-rag-benchmark: hybrid retrieval over legal corpora"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
---

# Abstract

We evaluate four retrieval recipes on the LegalBench-RAG benchmark: BM25, dense (BGE-small +
FAISS), reciprocal rank fusion of the two, and cross-encoder reranking on top of either. We
report nDCG, recall, MRR, and MAP at k = 10 across four legal sub-corpora (contractnli, cuad,
maud, privacy_qa) and discuss when each recipe pays off in throughput-adjusted terms.

# Method

BM25 uses rank-bm25 with a regex-based word tokenizer (lowercased). The dense retriever encodes
documents and queries with BAAI/bge-small-en-v1.5 and stores vectors in a FAISS inner-product
flat index (cosine similarity via L2 normalization). The hybrid is RRF (Cormack et al., 2009)
with k=60 over the top-200 of each base retriever. The reranker is cross-encoder/ms-marco-MiniLM-
L-6-v2, applied to the top-50 of any base retriever.

# Datasets

LegalBench-RAG (Pipitone & Alami, 2024) ships four sub-corpora with paired queries and span-
level ground truth. We collapse spans to doc-level qrels.

# Results

To be filled in from `results/SUMMARY.md` once the runs complete on the target machine.

# Limitations

- Doc-level qrels only.
- Cross-encoder reranking is single-model; an LLM-as-judge pass is out of scope.
- No latency budget enforcement.

# References

See README.
