from __future__ import annotations

from pathlib import Path

from legal_rag_benchmark.data.loader import load_from_jsonl
from legal_rag_benchmark.retrievers.base import Hit, Retriever
from legal_rag_benchmark.retrievers.bm25 import BM25Retriever
from legal_rag_benchmark.retrievers.hybrid import RRFHybridRetriever

FIXTURES = Path(__file__).parent / "fixtures"


class _StubRetriever(Retriever):
    """Hand-coded ranked list so the RRF math is deterministic."""

    def __init__(self, name: str, ranking: list[str]) -> None:
        self.name = name
        self._ranking = ranking

    def index(self, documents: list) -> None:
        pass

    def search(self, query: str, top_k: int) -> list[Hit]:
        return [
            Hit(doc_id=d, score=1.0 - i * 0.1, rank=i + 1)
            for i, d in enumerate(self._ranking[:top_k])
        ]


def test_rrf_combines_two_rankings() -> None:
    # bm25 likes [a, b, c]; dense likes [c, a, d]. With RRF k=60 the doc that
    # appears in both at decent rank should beat one-list-only docs.
    a = _StubRetriever("a", ["a", "b", "c"])
    b = _StubRetriever("b", ["c", "a", "d"])
    fusion = RRFHybridRetriever([a, b], k=60)
    fusion.index([])
    top = fusion.search("any", top_k=5)
    ids = [h.doc_id for h in top]
    # 'a' and 'c' are in both lists; both should appear before 'd' (only in b)
    assert ids.index("a") < ids.index("d")
    assert ids.index("c") < ids.index("d")


def test_rrf_requires_two_retrievers() -> None:
    try:
        RRFHybridRetriever([_StubRetriever("a", ["x"])])
    except ValueError:
        return
    raise AssertionError("expected ValueError for single retriever")


def test_hybrid_on_tiny_corpus() -> None:
    bundle = load_from_jsonl(FIXTURES, "tiny")
    bm25 = BM25Retriever()
    # second retriever: a different BM25 with extreme b (length normalization)
    # to give RRF something to fuse. (full dense test is gated behind 'slow'.)
    other = BM25Retriever()
    other.config = bm25.config.model_copy(update={"b": 0.1})
    fusion = RRFHybridRetriever([bm25, other])
    fusion.index(bundle.documents)
    hits = fusion.search(bundle.queries[0].text, top_k=5)
    assert len(hits) == 5
    assert hits[0].rank == 1
