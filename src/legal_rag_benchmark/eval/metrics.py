"""IR metrics: nDCG@k, MRR@k, Recall@k, MAP.

Implementations follow Manning, Raghavan & Schutze (Intro to IR, 2008) and
the TREC pytrec_eval reference where convenient. We intentionally don't
pull pytrec_eval as a dep so the package stays pure Python + numpy and
installs cleanly on any platform.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..retrievers.base import Hit


@dataclass
class QueryResult:
    qid: str
    hits: list[Hit]
    relevant: dict[str, int]  # qrels: doc_id -> grade (>=1 is relevant)


def recall_at_k(result: QueryResult, k: int) -> float:
    relevant_ids = {d for d, g in result.relevant.items() if g > 0}
    if not relevant_ids:
        return 0.0
    retrieved = {h.doc_id for h in result.hits[:k]}
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(result: QueryResult, k: int) -> float:
    relevant_ids = {d for d, g in result.relevant.items() if g > 0}
    for i, h in enumerate(result.hits[:k], start=1):
        if h.doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(result: QueryResult, k: int) -> float:
    relevant_ids = {d for d, g in result.relevant.items() if g > 0}
    if not relevant_ids:
        return 0.0
    hits_count = 0
    precision_sum = 0.0
    for i, h in enumerate(result.hits[:k], start=1):
        if h.doc_id in relevant_ids:
            hits_count += 1
            precision_sum += hits_count / i
    return precision_sum / min(len(relevant_ids), k)


def ndcg_at_k(result: QueryResult, k: int) -> float:
    """nDCG using graded relevance (binary if grades are all 1)."""
    if not result.relevant:
        return 0.0
    gains = [result.relevant.get(h.doc_id, 0) for h in result.hits[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_grades = sorted(result.relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def evaluate(
    results: Sequence[QueryResult],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Macro-average each metric across queries. Returns a flat dict like
    {"ndcg@10": 0.671, "mrr@10": 0.745, ...}.
    """
    out: dict[str, float] = {}
    n = len(results) or 1
    for k in k_values:
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in results) / n
        out[f"recall@{k}"] = sum(recall_at_k(r, k) for r in results) / n
        out[f"mrr@{k}"] = sum(reciprocal_rank(r, k) for r in results) / n
        out[f"map@{k}"] = sum(average_precision(r, k) for r in results) / n
    return out
