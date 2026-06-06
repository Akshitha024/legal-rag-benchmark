"""Cross-encoder reranker stage.

Reranking is a wrapper, not a standalone retriever: it takes the top-N from
some base retriever and rescores (query, document) pairs with a cross
encoder. MiniLM-L-6-v2 is small enough to use on CPU; it's the obvious
starting point. A bigger model (BGE reranker, Cohere rerank-3.5) usually
helps but costs more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import RerankConfig
from ..data.loader import Document
from .base import Hit, Retriever

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class RerankingRetriever(Retriever):
    name = "rerank"

    def __init__(
        self,
        base: Retriever,
        config: RerankConfig | None = None,
    ) -> None:
        self.base = base
        self.config = config or RerankConfig()
        self._reranker: CrossEncoder | None = None
        self._doc_text: dict[str, str] = {}
        self.name = f"rerank({base.name})"

    def _load(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(self.config.model_name)
        return self._reranker

    def index(self, documents: list[Document]) -> None:
        self.base.index(documents)
        self._doc_text = {d.doc_id: d.text for d in documents}

    def search(self, query: str, top_k: int) -> list[Hit]:
        base_k = max(self.config.top_n_to_rerank, top_k)
        candidates = self.base.search(query, base_k)
        if not candidates:
            return []
        pairs = [(query, self._doc_text[h.doc_id]) for h in candidates]
        scores = self._load().predict(pairs, batch_size=self.config.batch_size)
        rescored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]
        return [
            Hit(doc_id=h.doc_id, score=float(s), rank=rank + 1)
            for rank, (h, s) in enumerate(rescored)
        ]
