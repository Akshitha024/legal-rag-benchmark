from __future__ import annotations

import math

from legal_rag_benchmark.eval.metrics import (
    QueryResult,
    average_precision,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from legal_rag_benchmark.retrievers.base import Hit


def _hits(doc_ids: list[str]) -> list[Hit]:
    return [Hit(doc_id=d, score=1.0 - i * 0.1, rank=i + 1) for i, d in enumerate(doc_ids)]


def test_recall_basic() -> None:
    r = QueryResult(qid="q", hits=_hits(["a", "b", "c"]), relevant={"a": 1, "d": 1})
    assert recall_at_k(r, 5) == 0.5
    assert recall_at_k(r, 1) == 0.5  # 1 of 2 relevant found at k=1


def test_recall_no_qrels() -> None:
    r = QueryResult(qid="q", hits=_hits(["a", "b"]), relevant={})
    assert recall_at_k(r, 5) == 0.0


def test_mrr_first_hit() -> None:
    r = QueryResult(qid="q", hits=_hits(["x", "y", "z"]), relevant={"y": 1})
    assert reciprocal_rank(r, 10) == 0.5  # found at rank 2


def test_mrr_not_found() -> None:
    r = QueryResult(qid="q", hits=_hits(["a", "b"]), relevant={"z": 1})
    assert reciprocal_rank(r, 10) == 0.0


def test_average_precision() -> None:
    # qrels: {a, c}; ranked: [a, x, c, y] -> precisions at 1, 3 = 1.0, 2/3 ; AP = (1.0 + 2/3) / 2
    r = QueryResult(qid="q", hits=_hits(["a", "x", "c", "y"]), relevant={"a": 1, "c": 1})
    expected = (1.0 + 2 / 3) / 2
    assert math.isclose(average_precision(r, 10), expected, rel_tol=1e-9)


def test_ndcg_perfect_ranking() -> None:
    r = QueryResult(qid="q", hits=_hits(["a", "b"]), relevant={"a": 1, "b": 1})
    assert math.isclose(ndcg_at_k(r, 10), 1.0)


def test_ndcg_worst_ranking() -> None:
    # qrels are {a, b}; we return [x, y, a, b]; ideal is [a, b, ...]
    r = QueryResult(
        qid="q",
        hits=_hits(["x", "y", "a", "b"]),
        relevant={"a": 1, "b": 1},
    )
    score = ndcg_at_k(r, 4)
    assert 0 < score < 1


def test_ndcg_graded() -> None:
    r = QueryResult(
        qid="q",
        hits=_hits(["a", "b", "c"]),
        relevant={"a": 1, "b": 3, "c": 2},
    )
    # not the ideal ordering (ideal: b, c, a), so nDCG < 1
    assert ndcg_at_k(r, 3) < 1.0


def test_evaluate_aggregates() -> None:
    rs = [
        QueryResult(qid="q1", hits=_hits(["a", "b"]), relevant={"a": 1}),
        QueryResult(qid="q2", hits=_hits(["x", "y"]), relevant={"y": 1}),
    ]
    out = evaluate(rs, [1, 5])
    assert "ndcg@1" in out
    assert "recall@5" in out
    assert "mrr@5" in out
    # q1 finds at rank 1 (rr=1), q2 finds at rank 2 (rr=0.5) -> mean 0.75
    assert math.isclose(out["mrr@5"], 0.75)
