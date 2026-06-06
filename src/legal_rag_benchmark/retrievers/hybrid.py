"""Reciprocal Rank Fusion of multiple retrievers.

RRF (Cormack, Clarke & Buettcher, 2009) doesn't need score normalization,
which is why it tends to be the safe default when combining heterogeneous
scorers like BM25 and a dense model. The k constant is the standard 60;
larger k flattens the contribution of low-ranked items.
"""

from __future__ import annotations

from collections import defaultdict

from ..data.loader import Document
from .base import Hit, Retriever


class RRFHybridRetriever(Retriever):
    name = "hybrid_rrf"

    def __init__(self, retrievers: list[Retriever], k: int = 60) -> None:
        if len(retrievers) < 2:
            raise ValueError("need at least two retrievers for fusion")
        self.retrievers = retrievers
        self.k = k
        # set a more readable composite name
        self.name = "rrf(" + "+".join(r.name for r in retrievers) + ")"

    def index(self, documents: list[Document]) -> None:
        for r in self.retrievers:
            r.index(documents)

    def search(self, query: str, top_k: int) -> list[Hit]:
        # over-fetch from each base retriever; RRF is sensitive to recall depth
        base_k = max(top_k * 4, 50)
        rrf_scores: dict[str, float] = defaultdict(float)
        for r in self.retrievers:
            hits = r.search(query, base_k)
            for h in hits:
                rrf_scores[h.doc_id] += 1.0 / (self.k + h.rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            Hit(doc_id=doc_id, score=score, rank=rank + 1)
            for rank, (doc_id, score) in enumerate(ranked)
        ]
