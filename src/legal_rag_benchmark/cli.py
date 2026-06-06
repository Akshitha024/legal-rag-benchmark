from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tabulate import tabulate

from .config import BM25Config, DenseConfig, EvalConfig, RerankConfig
from .data.loader import ALL_CORPORA, fetch_legalbench_rag, load_from_jsonl
from .eval.runner import run_retriever, write_results
from .retrievers.base import Retriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever
from .retrievers.hybrid import RRFHybridRetriever
from .retrievers.rerank import RerankingRetriever

app = typer.Typer(add_completion=False, help="legal-rag-benchmark CLI")
data_app = typer.Typer(help="dataset operations")
eval_app = typer.Typer(help="run evaluations")
report_app = typer.Typer(help="report generation")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")


def _build_retrievers(
    names: list[str],
    bm25_config: BM25Config,
    dense_config: DenseConfig,
    rerank_config: RerankConfig,
) -> list[Retriever]:
    pool: dict[str, Retriever] = {}

    def bm25() -> BM25Retriever:
        if "bm25" not in pool:
            pool["bm25"] = BM25Retriever(bm25_config)
        return pool["bm25"]  # type: ignore[return-value]

    def dense() -> DenseRetriever:
        if "dense" not in pool:
            pool["dense"] = DenseRetriever(dense_config)
        return pool["dense"]  # type: ignore[return-value]

    out: list[Retriever] = []
    for n in names:
        if n == "bm25":
            out.append(bm25())
        elif n == "dense":
            out.append(dense())
        elif n == "hybrid":
            out.append(RRFHybridRetriever([bm25(), dense()]))
        elif n == "rerank_dense":
            out.append(RerankingRetriever(dense(), rerank_config))
        elif n == "rerank_hybrid":
            out.append(
                RerankingRetriever(
                    RRFHybridRetriever([bm25(), dense()]),
                    rerank_config,
                )
            )
        else:
            raise typer.BadParameter(f"unknown retriever: {n}")
    return out


@data_app.command("prepare")
def data_prepare(
    corpus: Annotated[
        list[str] | None,
        typer.Option(help="corpus name; pass multiple times (defaults to all four)"),
    ] = None,
    dest: Annotated[Path, typer.Option(help="destination directory")] = Path("data/processed"),
) -> None:
    """Download a LegalBench-RAG sub-corpus and convert to our JSONL schema."""
    targets = corpus if corpus else list(ALL_CORPORA)
    for c in targets:
        fetch_legalbench_rag(c, dest)  # type: ignore[arg-type]


@eval_app.command("run")
def eval_run(
    corpus: Annotated[str, typer.Option(help="corpus name")] = "contractnli",
    data_dir: Annotated[Path, typer.Option(help="processed data dir")] = Path("data/processed"),
    retrievers: Annotated[
        list[str] | None,
        typer.Option(
            help="retrievers to run; choose from bm25, dense, hybrid, rerank_dense, rerank_hybrid"
        ),
    ] = None,
    topk: Annotated[int, typer.Option(help="primary k for table")] = 10,
    out_dir: Annotated[Path, typer.Option(help="results directory")] = Path("results"),
) -> None:
    bundle = load_from_jsonl(data_dir, corpus)
    eval_config = EvalConfig()
    if topk not in eval_config.k_values:
        eval_config.k_values.append(topk)
        eval_config.k_values.sort()
    chosen = retrievers if retrievers else ["bm25", "dense", "hybrid"]
    runners = _build_retrievers(
        chosen,
        BM25Config(),
        DenseConfig(),
        RerankConfig(),
    )

    rows: list[list[str | float]] = []
    for r in runners:
        metrics, per_query, _, _ = run_retriever(r, bundle, eval_config.k_values)
        write_results(out_dir, corpus, r.name, metrics, per_query)
        rows.append(
            [
                r.name,
                metrics[f"ndcg@{topk}"],
                metrics[f"recall@{topk}"],
                metrics[f"mrr@{topk}"],
                metrics[f"map@{topk}"],
                metrics["queries_per_sec"],
            ]
        )

    print()
    print(
        tabulate(
            rows,
            headers=[
                "retriever",
                f"nDCG@{topk}",
                f"Recall@{topk}",
                f"MRR@{topk}",
                f"MAP@{topk}",
                "QPS",
            ],
            floatfmt=".3f",
            tablefmt="github",
        )
    )


@report_app.command("build")
def report_build(
    results_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    """Collate every <corpus>__<retriever>__metrics.json into a markdown table."""
    rows: list[dict[str, str | float]] = []
    for f in sorted(results_dir.glob("*__metrics.json")):
        # filename layout is corpus__retriever__metrics.json
        stem = f.stem.removesuffix("__metrics")
        if "__" not in stem:
            continue
        corpus, retriever = stem.split("__", 1)
        metrics = json.loads(f.read_text())
        rows.append({"corpus": corpus, "retriever": retriever, **metrics})

    if not rows:
        logger.warning("no metrics files found under {}", results_dir)
        return

    md_lines = ["# Results", ""]
    ks = sorted({int(k.split("@")[1]) for r in rows for k in r if "@" in str(k)})
    headers = ["corpus", "retriever", *[f"nDCG@{k}" for k in ks]]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells: list[str] = [str(r["corpus"]), str(r["retriever"])]
        cells.extend(f"{float(r.get(f'ndcg@{k}', 0)):.3f}" for k in ks)
        md_lines.append("| " + " | ".join(cells) + " |")
    out = results_dir / "SUMMARY.md"
    out.write_text("\n".join(md_lines) + "\n")
    logger.info("wrote {}", out)


if __name__ == "__main__":
    app()
