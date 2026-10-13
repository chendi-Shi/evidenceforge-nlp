"""Explicit stages make expensive work and evaluation boundaries visible."""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Scientific evidence retrieval and reproducible ranking experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download the checksum-verified public BEIR SciFact data")
    build = commands.add_parser("index", help="Build retrieval index and pinned neural embedding cache")
    build.add_argument("--lexical-only", action="store_true")
    experiment = commands.add_parser("experiment", help="Train/validate on train only, then evaluate fixed test")
    experiment.add_argument("--output-dir", type=Path, default=Path("results/scifact"))
    adapt = commands.add_parser("adapt", help="Train real CPU LoRA with matched negative controls and evaluate transfer")
    adapt.add_argument("--stage", choices=["prepare", "train", "evaluate", "all"], default="all")
    adapt.add_argument("--output-dir", type=Path, default=Path("results/adaptation"))
    adapt.add_argument("--transfer-dir", type=Path, default=Path("data/nfcorpus"))
    adapt.add_argument("--epochs", type=int, default=2)
    adapt.add_argument("--max-train-queries", type=int, default=10000)
    search = commands.add_parser("search", help="Retrieve relevant scientific abstracts")
    search.add_argument("query")
    search.add_argument("--method", choices=["bm25", "dense", "rrf", "ltr"], default="rrf")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--model-dir", type=Path, default=Path("results/scifact/model"))
    search.add_argument("--adapter-dir", type=Path, help="Use a trained LoRA encoder for dense/RRF search")
    serve = commands.add_parser("serve", help="Serve search API and OpenAPI documentation at /docs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-dir", type=Path, default=Path("results/scifact/model"))
    serve.add_argument("--lexical-only", action="store_true")
    serve.add_argument("--adapter-dir", type=Path, help="Serve a trained LoRA encoder; v1 LambdaMART is not loaded")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be between 1 and 64")
    if args.command == "search":
        if args.adapter_dir is not None and args.method not in {"dense", "rrf"}:
            parser.error("--adapter-dir requires --method dense or rrf")
        if not re.search(r"\w", args.query) or len(args.query) > 2000:
            parser.error("query must contain words or numbers and at most 2000 characters")
        if not 1 <= args.top_k <= 50:
            parser.error("--top-k must be between 1 and 50")
    if args.command == "download":
        from evidenceforge.data import download_scifact
        download_scifact(args.data_dir)
        print(f"Data ready: {args.data_dir.resolve()}")
        return
    if args.command == "experiment":
        from evidenceforge.experiment import run_experiment
        report = run_experiment(args.data_dir, args.cache_dir, args.output_dir, threads=args.threads)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "adapt":
        from evidenceforge.neural_experiment import run_adaptation_experiment
        report = run_adaptation_experiment(args.data_dir, args.transfer_dir, args.cache_dir, args.output_dir,
                                          stage=args.stage, epochs=args.epochs,
                                          max_train_queries=args.max_train_queries, threads=args.threads)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "serve":
        if args.adapter_dir is not None and args.lexical_only:
            parser.error("--adapter-dir cannot be combined with --lexical-only")
        import uvicorn
        from evidenceforge.api import create_app
        app = create_app(args.data_dir, args.cache_dir, args.model_dir,
                         dense=not args.lexical_only, threads=args.threads, adapter_dir=args.adapter_dir)
        uvicorn.run(app, host=args.host, port=args.port)
        return
    from evidenceforge.data import load_dataset
    from evidenceforge.retrieval import RetrievalIndex
    dense = args.method != "bm25" if args.command == "search" else not args.lexical_only
    encoder = None
    if args.command == "search" and args.adapter_dir is not None:
        from evidenceforge.adaptation import AdaptedEncoder
        from evidenceforge.model_assets import download_float_minilm
        encoder = AdaptedEncoder.load(download_float_minilm(args.cache_dir), args.adapter_dir, threads=args.threads)
    index = RetrievalIndex(load_dataset(args.data_dir), args.cache_dir, dense=dense, threads=args.threads, encoder=encoder)
    index.build()
    if args.command == "index":
        print(json.dumps(index.metadata, ensure_ascii=False, indent=2))
        return
    if args.method == "ltr":
        from evidenceforge.ranker import EvidenceRanker
        ranker = EvidenceRanker.load(args.model_dir)
        ranker.validate_index(index.metadata)
        hits = ranker.rerank(args.query, index.search(args.query, k=50))
    else:
        hits = index.search(args.query, method=args.method, k=args.top_k)
    print(json.dumps({"query": args.query, "method": args.method,
                      "notice": "Relevance does not establish support, contradiction, or medical validity.",
                      "results": hits[:args.top_k]}, ensure_ascii=False, indent=2))
