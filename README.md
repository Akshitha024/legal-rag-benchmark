# legal-rag-benchmark

Retrieval evaluation harness for legal corpora. Runs BM25, dense (sentence-transformers + FAISS),
RRF hybrid, and cross-encoder reranking against the four LegalBench-RAG sub-corpora and reports
nDCG@k, Recall@k, MRR@k, and MAP@k.

The point of this repo is to have a single place to ask: for a given legal corpus, which retrieval
recipe is actually best, and how much do the extra stages cost? Most "use a vector database"
write-ups skip the comparison; this one runs it.

## What's in here

- BM25 (rank-bm25 with custom tokenization)
- Dense retrieval with BGE-small-en-v1.5 + FAISS inner product
- RRF hybrid fusion (Cormack et al., 2009)
- Cross-encoder reranker (MiniLM-L-6) over the top-N of any base retriever
- Eval: nDCG@k, Recall@k, MRR@k, MAP@k with index-time and search-time metrics

The benchmark target is [LegalBench-RAG](https://arxiv.org/abs/2408.10343) (Pipitone &
Alami, 2024), four legal sub-corpora with paired queries and span-level ground truth.
We collapse spans to doc-level qrels for now (span-level eval is in the TODOs).

| Sub-corpus    | Docs | Queries | Domain                  |
|---------------|-----:|--------:|-------------------------|
| `contractnli` |  607 |   2,091 | NDA clauses             |
| `cuad`        |  510 |   4,042 | Commercial contracts    |
| `maud`        |  152 |     187 | M&A understanding       |
| `privacy_qa`  |    8 |   1,750 | Privacy policies        |

## Quickstart

```bash
# install (uses uv)
make install

# download + prep one corpus
uv run lrb data prepare --corpus contractnli --dest data/processed

# build indices and run eval
uv run lrb eval run --corpus contractnli --retrievers bm25 dense hybrid --topk 10

# also try with reranking (slower)
uv run lrb eval run --corpus contractnli --retrievers rerank_dense rerank_hybrid --topk 10

# collate per-corpus results into a summary markdown
uv run lrb report build
```

For a smoke test that does not touch the network:

```bash
uv run pytest -q
```

## Results

Numbers below are from the standard run (top-k=10, default config). Update by re-running the
commands in the Quickstart and committing `results/SUMMARY.md`.

> **Status (2026-06-06):** The harness is wired and tested on a tiny in-repo fixture. Full
> LegalBench-RAG runs are queued for a clean machine. The table below will be filled in from
> `results/SUMMARY.md` once the runs complete.

```text
| corpus      | retriever          | nDCG@10 | Recall@10 | MRR@10 | MAP@10 |
| contractnli | bm25               |   TBD   |    TBD    |  TBD   |  TBD   |
| contractnli | dense              |   TBD   |    TBD    |  TBD   |  TBD   |
| contractnli | rrf(bm25+dense)    |   TBD   |    TBD    |  TBD   |  TBD   |
| contractnli | rerank(rrf(bm25+d) |   TBD   |    TBD    |  TBD   |  TBD   |
| ...
```

What we expect to see (and what published LegalBench-RAG numbers suggest):

1. BM25 is a tough baseline on legal text. Word overlap matters; legal queries are
   surprisingly literal.
2. Dense retrieval helps most on `privacy_qa` (paraphrase-heavy) and least on `contractnli`
   (clause boundaries are lexically distinctive).
3. RRF hybrid beats either pure approach on the macro average.
4. Reranking pays off most when the base recall is already high (i.e. on top of hybrid).

We will know whether those hold once the table is filled in.

## Design notes

- **Why RRF instead of weighted score fusion?** Score normalization across BM25 and a dense model
  is fragile in practice. RRF only needs ranks, which makes it the safe default when you do not
  want to tune a fusion weight per corpus.
- **Why FAISS Flat instead of HNSW/IVF?** For corpora under a million documents the index-build
  cost of an ANN structure dominates and the recall is worse. Flat IP is exact and fast enough.
- **Why BGE-small?** It is the smallest sentence encoder I have seen punch above its weight on
  BEIR. It fits on CPU. Swap via `DenseConfig(model_name=...)` if you have a GPU and want to try
  `bge-large` or a legal-domain model.
- **Why doc-level qrels?** LegalBench-RAG ships span-level annotations; doing span-level
  eval well needs careful handling of doc-spanning answers. Doc-level is a faithful upper bound
  and lets us compare retrievers without that complication first.

## Known limitations

- Doc-level qrels only (span-level is the next item on the list).
- Dense retriever loads the whole encoder model into memory; for very large corpora on a small
  laptop you will want to switch to streaming encode + on-disk index.
- The reranker uses a single cross-encoder; an LLM-as-judge rerank is out of scope for this repo.
- No latency budget enforcement. The `queries_per_sec` column is informational only.

## What's next

- [ ] Span-level evaluation, matching the original LegalBench-RAG protocol
- [ ] Add ColBERT-style late interaction as a fourth base retriever
- [ ] Domain-specific encoders (Legal-BERT, SaulLM embeddings) once their license terms are clear
- [ ] Caching layer for query embeddings so repeated runs do not re-encode
- [ ] Plot scripts for nDCG vs. k and recall-precision curves

## References

- Pipitone, N., & Alami, G. (2024). *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain.* arXiv:2408.10343
- Cormack, G., Clarke, C., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR.
- Manning, C., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval.* CUP.
- Xiao, S., et al. (2024). *C-Pack: Packed Resources For General Chinese Embeddings.* SIGIR. (BGE family)
- Henderson, P., et al. (2022). *Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset.* NeurIPS.

## Citation

```bibtex
@software{lingampally2026legalragbench,
  author  = {Lingampally, Akshitha Reddy},
  title   = {legal-rag-benchmark: Hybrid retrieval evaluation on legal corpora},
  year    = {2026},
  url     = {https://github.com/Akshitha024/legal-rag-benchmark}
}
```

## License

MIT. See [LICENSE](./LICENSE).
