from __future__ import annotations

from pathlib import Path

from legal_rag_benchmark.data.loader import load_from_jsonl
from legal_rag_benchmark.eval.runner import run_retriever
from legal_rag_benchmark.retrievers.bm25 import BM25Retriever, tokenize

FIXTURES = Path(__file__).parent / "fixtures"


def test_tokenize_lowercase() -> None:
    assert tokenize("Hello WORLD! 123 a") == ["hello", "world", "123"]


def test_tokenize_keeps_alphanumeric() -> None:
    out = tokenize("Section 12.3(b)")
    assert "section" in out
    # rank-bm25 doesn't know what to do with numeric tokens, but we keep them
    assert "12" in out or "3" in out


def test_bm25_finds_relevant_on_tiny() -> None:
    bundle = load_from_jsonl(FIXTURES, "tiny")
    r = BM25Retriever()
    metrics, _, _, _ = run_retriever(r, bundle, k_values=[1, 5])
    # 4 queries, all should have at least one relevant in top-5 for this small corpus
    assert metrics["recall@5"] > 0.5
    assert metrics["mrr@5"] > 0
