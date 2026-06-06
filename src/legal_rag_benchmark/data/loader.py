"""Loaders for the benchmark corpora.

The primary benchmark target is LegalBench-RAG (Pipitone & Alami, 2024). It
ships four legal sub-corpora with paired query/relevant-span ground truth:

  - contractnli   (607 docs, 2091 queries)  - NDA clause retrieval
  - cuad          (510 docs, 4042 queries)  - contract clause QA
  - maud          (152 docs, 187 queries)   - M&A understanding
  - privacy_qa    (8 docs, 1750 queries)    - privacy policy QA

For first-pass smoke tests and CI we ship a tiny in-repo fixture (see
``tests/fixtures``) so the test suite does not touch the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

Corpus = Literal["contractnli", "cuad", "maud", "privacy_qa"]
ALL_CORPORA: tuple[Corpus, ...] = ("contractnli", "cuad", "maud", "privacy_qa")


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    # qrels[doc_id] = relevance grade (0 = irrelevant, 1+ = relevant)
    relevant: dict[str, int]


@dataclass
class CorpusBundle:
    name: str
    documents: list[Document]
    queries: list[Query]

    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]


def load_from_jsonl(corpus_dir: Path, name: str) -> CorpusBundle:
    """Load a corpus from a directory of {corpus}.docs.jsonl + {corpus}.queries.jsonl.

    Expected schema:
      docs.jsonl     : {"doc_id": str, "text": str, "title": str?}
      queries.jsonl  : {"qid": str, "text": str, "relevant": {doc_id: grade}}
    """
    docs_path = corpus_dir / f"{name}.docs.jsonl"
    queries_path = corpus_dir / f"{name}.queries.jsonl"
    if not docs_path.exists() or not queries_path.exists():
        raise FileNotFoundError(f"missing {docs_path.name} or {queries_path.name} in {corpus_dir}")

    documents = [
        Document(
            doc_id=row["doc_id"],
            text=row["text"],
            title=row.get("title"),
        )
        for row in _iter_jsonl(docs_path)
    ]
    queries = [
        Query(
            qid=row["qid"],
            text=row["text"],
            relevant=row["relevant"],
        )
        for row in _iter_jsonl(queries_path)
    ]
    logger.info(
        "loaded corpus '{}': {} documents, {} queries",
        name,
        len(documents),
        len(queries),
    )
    return CorpusBundle(name=name, documents=documents, queries=queries)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def fetch_legalbench_rag(corpus: Corpus, dest: Path) -> Path:
    """Download a LegalBench-RAG sub-corpus and convert to our JSONL format.

    We pull from the public HuggingFace mirror. The conversion strips the
    original span-level annotations down to doc-level qrels (relevant doc =
    any doc with a positive span). Span-level eval is a planned follow-up.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "fetch_legalbench_rag requires the `datasets` package; "
            "install with `uv sync` or `pip install datasets`"
        ) from e

    dest.mkdir(parents=True, exist_ok=True)
    docs_path = dest / f"{corpus}.docs.jsonl"
    queries_path = dest / f"{corpus}.queries.jsonl"

    if docs_path.exists() and queries_path.exists():
        logger.info("corpus '{}' already prepared at {}", corpus, dest)
        return dest

    # zeroentropy/LegalBench-RAG is the canonical HF mirror. Schema details
    # are at https://huggingface.co/datasets/zeroentropy/LegalBench-RAG.
    logger.info("downloading LegalBench-RAG/{} from HuggingFace", corpus)
    ds = load_dataset("zeroentropy/LegalBench-RAG", corpus)

    # The dataset exposes 'corpus' and 'qrels' splits. Different mirror
    # versions name things slightly differently, so we look for both.
    doc_split = ds.get("corpus") or ds.get("documents") or ds.get("train")
    qrel_split = ds.get("qrels") or ds.get("queries") or ds.get("test")
    if doc_split is None or qrel_split is None:
        raise RuntimeError(
            f"could not find document/query splits in dataset; got keys: {list(ds.keys())}"
        )

    with docs_path.open("w") as f:
        for row in doc_split:
            f.write(json.dumps(_normalize_doc(row)) + "\n")
    with queries_path.open("w") as f:
        for row in qrel_split:
            f.write(json.dumps(_normalize_query(row)) + "\n")

    logger.info("wrote {} and {}", docs_path.name, queries_path.name)
    return dest


def _normalize_doc(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": str(row.get("doc_id") or row.get("id") or row["_id"]),
        "text": row.get("text") or row.get("content") or row["body"],
        "title": row.get("title"),
    }


def _normalize_query(row: dict[str, Any]) -> dict[str, Any]:
    relevant: dict[str, int]
    if "relevant" in row and isinstance(row["relevant"], dict):
        relevant = {str(k): int(v) for k, v in row["relevant"].items()}
    elif "relevant_docs" in row:
        relevant = {str(d): 1 for d in row["relevant_docs"]}
    else:
        # span-level: collapse spans to their parent doc
        relevant = {}
        for span in row.get("relevant_spans", []):
            relevant[str(span["doc_id"])] = max(relevant.get(str(span["doc_id"]), 0), 1)
    return {
        "qid": str(row.get("qid") or row.get("query_id") or row["id"]),
        "text": row.get("text") or row.get("query"),
        "relevant": relevant,
    }
