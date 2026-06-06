from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Paths(BaseModel):
    root: Path = Field(default_factory=lambda: Path.cwd())

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def indices(self) -> Path:
        return self.root / "indices"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        for p in (self.data_raw, self.data_processed, self.indices, self.results):
            p.mkdir(parents=True, exist_ok=True)


class DenseConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    normalize: bool = True
    max_seq_length: int = 512


class RerankConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    top_n_to_rerank: int = 50


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    # rank-bm25's tokenizer is naive; we lowercase + split on word boundaries
    lowercase: bool = True


class EvalConfig(BaseModel):
    k_values: list[int] = [1, 3, 5, 10, 20]
    primary_k: int = 10
