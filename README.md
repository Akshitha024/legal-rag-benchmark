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

First-pass numbers from CUAD-QA (theatticusproject/cuad-qa), 1,000 queries × 36 unique
contracts (the queries cap at 1K so the run finishes on a laptop; a full sweep is queued).
All metrics are macro-averaged across queries; QPS is queries-per-second measured end to end.

| corpus | retriever         | nDCG@10 | Recall@10 | MRR@10 | MAP@10 |   QPS |
|--------|-------------------|--------:|----------:|-------:|-------:|------:|
| cuad   | bm25              |   0.245 |     0.493 |  0.171 |  0.171 |  4948 |
| cuad   | dense (bge-small) |   0.133 |     0.301 |  0.084 |  0.084 |   135 |
| cuad   | rrf(bm25+dense)   |   0.230 |     0.459 |  0.161 |  0.161 |   159 |

A few things from this run that are worth being honest about:

1. **BM25 wins on CUAD.** That is not the textbook result. The reason here is the corpus
   characteristics: each CUAD contract is 30K to 70K characters, and the dense encoder
   (BGE-small-en, 512-token cap) only sees the first ~2,000 chars of each contract. The
   tail of the document is invisible to dense. BM25 indexes the whole text.
2. **Hybrid does not rescue dense.** Same root cause. RRF gives equal weight to both base
   rankings, so a noisy dense ranking drags the fusion below pure BM25. This is exactly
   the failure mode the RRF authors warn about when one base is much weaker than the other.
3. **The fix for dense is chunked indexing.** Split each contract into ~512-token chunks,
   index every chunk, then aggregate chunk scores back to doc-level (max or sum). That is
   the next item on the TODO list and the result we expect to flip the ranking.
4. **Recall@10 is around 50% for BM25 on this subset.** The CUAD questions are templated
   ("Highlight the parts of this contract related to ...") and share most of their text
   across contracts. The IR system has to disambiguate using the clause-type wording alone,
   which is a genuinely hard signal.

Reproduce with:

```bash
uv run lrb data prepare --corpus cuad --max-queries 1000
uv run lrb eval run --corpus cuad --retrievers bm25 --retrievers dense --retrievers hybrid --topk 10
uv run lrb report build
```

Full per-k breakdown lives in [`results/SUMMARY.md`](./results/SUMMARY.md).

LegalBench-RAG span-level numbers are still TODO. That benchmark's source documents are
hosted as a manual download on GitHub (`zeroentropy-ai/legalbenchrag`); wiring that fetcher
up properly is the next item.

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

- [ ] Chunked dense indexing (split contracts into 512-token chunks, aggregate scores back).
      This is the obvious win after the first results came in.
- [ ] LegalBench-RAG corpus fetcher (the source PDFs live on GitHub, not HF).
- [ ] Span-level evaluation, matching the original LegalBench-RAG protocol.
- [ ] Add ColBERT-style late interaction as a fourth base retriever.
- [ ] Domain-specific encoders (Legal-BERT, SaulLM embeddings) once their license terms are clear.
- [ ] Caching layer for query embeddings so repeated runs do not re-encode.
- [ ] Plot scripts for nDCG vs. k and recall-precision curves.

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


## Documentation and test artifacts

- Long-form research report (15-page target, in progress): [`docs/_report/research_report.md`](./docs/_report/research_report.md). Render to PDF with `make pdf` (requires `pandoc` + `xelatex`).
- Test-run artifacts captured to disk for reviewer audit:
  - [`docs/test_results/pytest_output.txt`](./docs/test_results/pytest_output.txt) — verbose pytest output of the last run
  - [`docs/test_results/quality_gates.txt`](./docs/test_results/quality_gates.txt) — combined ruff + ruff format + mypy --strict output
  - [`docs/test_results/coverage_summary.txt`](./docs/test_results/coverage_summary.txt) — pytest-cov summary
- Regenerate with `make test-artifacts`.

