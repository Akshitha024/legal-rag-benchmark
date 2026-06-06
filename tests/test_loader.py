from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag_benchmark.data.loader import load_from_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_tiny_corpus() -> None:
    bundle = load_from_jsonl(FIXTURES, "tiny")
    assert bundle.name == "tiny"
    assert len(bundle.documents) == 8
    assert len(bundle.queries) == 4
    assert bundle.documents[0].doc_id == "d1"
    assert "confidentiality" in bundle.documents[0].text


def test_load_missing_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_jsonl(tmp_path, "nonexistent")


def test_queries_have_qrels() -> None:
    bundle = load_from_jsonl(FIXTURES, "tiny")
    for q in bundle.queries:
        assert q.relevant, f"query {q.qid} has empty qrels"
        for grade in q.relevant.values():
            assert grade >= 1
