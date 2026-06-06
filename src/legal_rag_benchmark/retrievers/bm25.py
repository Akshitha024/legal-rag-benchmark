"""BM25 retriever, thin wrapper around rank-bm25.

rank-bm25 is unsophisticated about tokenization (it's a pure-Python list
operation), so we do our own preprocessing: lowercase + split on word
boundaries, drop tokens of length 1. For legal text this matters less than
for, say, biomedical text - section numbers and citations are usually
multi-character.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from ..config import BM25Config
from ..data.loader import Document
from .base import Hit, Retriever

if TYPE_CHECKING:
    pass

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 1]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()
        self._index: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    def index(self, documents: list[Document]) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        tokenized = [tokenize(d.text, self.config.lowercase) for d in documents]
        # rank-bm25 takes k1/b at construction; we re-instantiate so config sticks
        self._index = BM25Okapi(tokenized, k1=self.config.k1, b=self.config.b)

    def search(self, query: str, top_k: int) -> list[Hit]:
        if self._index is None:
            raise RuntimeError("call .index() before .search()")
        q_tokens = tokenize(query, self.config.lowercase)
        scores = self._index.get_scores(q_tokens)
        # argpartition is faster than full sort when top_k << N
        top_k = min(top_k, len(self._doc_ids))
        top_idx = scores.argsort()[-top_k:][::-1]
        return [
            Hit(doc_id=self._doc_ids[int(i)], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_idx)
        ]
