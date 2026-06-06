"""Dense retriever using sentence-transformers + FAISS inner product index.

We default to BGE-small-en-v1.5 (33M params, 384-dim) because it punches
above its weight on BEIR and runs on CPU in a reasonable time. Swap via
DenseConfig.model_name for larger models. We L2-normalize both doc and
query embeddings so inner product == cosine similarity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..config import DenseConfig
from ..data.loader import Document
from .base import Hit, Retriever

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, config: DenseConfig | None = None) -> None:
        self.config = config or DenseConfig()
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._doc_ids: list[str] = []

    def _load_model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name)
            self._model.max_seq_length = self.config.max_seq_length
        return self._model

    def _encode(self, texts: list[str]) -> NDArray[Any]:
        model = self._load_model()
        emb = model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return emb.astype(np.float32)

    def index(self, documents: list[Document]) -> None:
        import faiss

        self._doc_ids = [d.doc_id for d in documents]
        embeddings = self._encode([d.text for d in documents])
        # inner product on L2-normalized vectors = cosine; cheap and exact for our sizes
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, query: str, top_k: int) -> list[Hit]:
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        if self._index is None:
            raise RuntimeError("call .index() before .search()")
        q_emb = self._encode(queries)
        top_k = min(top_k, len(self._doc_ids))
        scores, idxs = self._index.search(q_emb, top_k)
        results: list[list[Hit]] = []
        for q_scores, q_idxs in zip(scores, idxs, strict=True):
            results.append(
                [
                    Hit(doc_id=self._doc_ids[int(i)], score=float(s), rank=rank + 1)
                    for rank, (i, s) in enumerate(zip(q_idxs, q_scores, strict=True))
                    if i >= 0  # faiss returns -1 for missing slots
                ]
            )
        return results
