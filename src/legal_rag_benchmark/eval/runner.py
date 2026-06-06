from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from ..data.loader import CorpusBundle
from ..retrievers.base import Retriever
from .metrics import QueryResult, evaluate


def run_retriever(
    retriever: Retriever,
    bundle: CorpusBundle,
    k_values: Iterable[int],
) -> tuple[dict[str, float], list[QueryResult], float, float]:
    """Index the corpus, run all queries, return (metrics, per-query, index_secs, search_secs)."""
    k_values = list(k_values)
    max_k = max(k_values)

    t0 = time.perf_counter()
    retriever.index(bundle.documents)
    index_secs = time.perf_counter() - t0
    logger.info("{} indexed in {:.2f}s", retriever.name, index_secs)

    t0 = time.perf_counter()
    per_query: list[QueryResult] = []
    for q in tqdm(bundle.queries, desc=f"search:{retriever.name}", leave=False):
        hits = retriever.search(q.text, max_k)
        per_query.append(QueryResult(qid=q.qid, hits=hits, relevant=q.relevant))
    search_secs = time.perf_counter() - t0

    metrics = evaluate(per_query, k_values)
    metrics["index_secs"] = index_secs
    metrics["search_secs"] = search_secs
    metrics["queries_per_sec"] = len(bundle.queries) / max(search_secs, 1e-6)
    return metrics, per_query, index_secs, search_secs


def write_results(
    out_dir: Path,
    corpus_name: str,
    retriever_name: str,
    metrics: dict[str, float],
    per_query: list[QueryResult],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / f"{corpus_name}__{retriever_name}__metrics.json"
    runs = out_dir / f"{corpus_name}__{retriever_name}__runs.jsonl"

    summary.write_text(json.dumps(metrics, indent=2))
    with runs.open("w") as f:
        for r in per_query:
            f.write(
                json.dumps(
                    {
                        "qid": r.qid,
                        "hits": [
                            {"doc_id": h.doc_id, "rank": h.rank, "score": h.score} for h in r.hits
                        ],
                    }
                )
                + "\n"
            )
    logger.info("wrote {} and {}", summary.name, runs.name)
    return summary
