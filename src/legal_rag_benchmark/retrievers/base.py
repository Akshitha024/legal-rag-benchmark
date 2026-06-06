from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.loader import Document


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    rank: int


class Retriever(ABC):
    name: str

    @abstractmethod
    def index(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[Hit]: ...

    def search_batch(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        # subclasses can override for vectorized search
        return [self.search(q, top_k) for q in queries]
