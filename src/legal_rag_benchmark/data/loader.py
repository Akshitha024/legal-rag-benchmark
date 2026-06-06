"""Loaders for the benchmark corpora.

Two real corpora are supported out of the box:

  - ``cuad``    : CUAD (Hendrycks et al., 2021) commercial contracts, derived
                  from theatticusproject/cuad-qa on HuggingFace. 510 unique
                  contracts, 22.4K clause-type questions. Doc-level qrels are
                  built by mapping each (question, contract) pair with a non-
                  empty answer span to ``{contract_title: 1}``.

  - ``legalbench_rag`` : The LegalBench-RAG benchmark (Pipitone & Alami,
                  2024) ships span-level annotations only; the source
                  documents must be downloaded separately from the project's
                  GitHub release. ``fetch_legalbench_rag`` documents the
                  manual step. Span-level eval is a planned follow-up.

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

Corpus = Literal["cuad", "contractnli", "maud", "privacy_qa"]
ALL_CORPORA: tuple[Corpus, ...] = ("cuad", "contractnli", "maud", "privacy_qa")


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


def fetch_cuad(
    dest: Path,
    split: str = "train",
    max_queries: int | None = None,
) -> Path:
    """Build a retrieval corpus + qrels from the CUAD-QA HuggingFace dataset.

    CUAD-QA (theatticusproject/cuad-qa) ships SQuAD-style rows; for each
    (contract, clause-type) pair it gives the question and either the
    extracted span or an empty answer. We collapse to:

      - corpus     : one document per unique contract title
      - queries    : one query per non-empty SQuAD row
      - qrels      : {contract_title: 1} for the contract the answer came from

    A single question often has the same wording across many contracts (the
    41 CUAD clause types are templated). That is a feature, not a bug: in
    practice an IR system has to disambiguate the question by the rest of the
    surrounding context. We keep qids unique by suffixing with the contract
    title so the eval is one-relevant-doc-per-query.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "fetch_cuad requires the `datasets` package; install with `uv sync`"
        ) from e

    dest.mkdir(parents=True, exist_ok=True)
    docs_path = dest / "cuad.docs.jsonl"
    queries_path = dest / "cuad.queries.jsonl"
    if docs_path.exists() and queries_path.exists():
        logger.info("cuad already prepared at {}", dest)
        return dest

    logger.info("loading CUAD-QA ({} split) from theatticusproject/cuad-qa", split)
    ds = load_dataset("theatticusproject/cuad-qa", split=split, trust_remote_code=True)

    seen_titles: dict[str, str] = {}
    queries_written = 0
    skipped_no_answer = 0

    with queries_path.open("w") as fq:
        for row in ds:
            title = str(row["title"]).strip()
            if not title:
                continue
            if title not in seen_titles:
                seen_titles[title] = str(row["context"])
            answers = row.get("answers") or {}
            texts = answers.get("text") if isinstance(answers, dict) else None
            if not texts or not any(t.strip() for t in texts):
                skipped_no_answer += 1
                continue
            qid = f"{row['id']}__{title[:40]}"
            fq.write(
                json.dumps(
                    {
                        "qid": qid,
                        "text": row["question"],
                        "relevant": {title: 1},
                    }
                )
                + "\n"
            )
            queries_written += 1
            if max_queries is not None and queries_written >= max_queries:
                break

    with docs_path.open("w") as fd:
        for title, ctx in seen_titles.items():
            fd.write(json.dumps({"doc_id": title, "text": ctx, "title": title}) + "\n")

    logger.info(
        "cuad prepared: {} docs, {} queries (skipped {} with empty answers)",
        len(seen_titles),
        queries_written,
        skipped_no_answer,
    )
    return dest


def fetch_legalbench_rag(corpus: Corpus, dest: Path) -> Path:
    """Stub for LegalBench-RAG sub-corpora.

    The original LegalBench-RAG benchmark (Pipitone & Alami, 2024) ships
    span-level QA pairs but the underlying corpora (the contracts, NDAs,
    M&A docs, privacy policies) live in the project's GitHub release at
    https://github.com/zeroentropy-ai/legalbenchrag and have to be
    downloaded manually. Until we wire that fetcher up, point this command
    at a directory containing the corresponding ``{corpus}.docs.jsonl`` +
    ``{corpus}.queries.jsonl`` files (the same schema as ``load_from_jsonl``
    consumes) and skip this step.
    """
    docs_path = dest / f"{corpus}.docs.jsonl"
    queries_path = dest / f"{corpus}.queries.jsonl"
    if docs_path.exists() and queries_path.exists():
        logger.info("found existing {}/{} prepared files in {}", corpus, corpus, dest)
        return dest
    raise NotImplementedError(
        f"LegalBench-RAG/{corpus} needs manual corpus setup: download from "
        "github.com/zeroentropy-ai/legalbenchrag and write to "
        f"{docs_path} and {queries_path}. See loader.py docstring for schema."
    )
