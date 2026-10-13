"""Supervised, candidate-controlled LambdaMART reranking for scientific evidence."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightgbm as lgb

from .evaluation import macro_metrics, query_metrics

SEED = 42
POOL_SIZE = 50
SCHEMA_VERSION = 1
GRID = ({"num_leaves": 7}, {"num_leaves": 15})
BASE_CONFIG = {"n_estimators": 100, "learning_rate": 0.05, "min_child_samples": 20, "reg_lambda": 1.0}
LEXICAL_FEATURES = (
    "query_token_recall", "document_token_precision", "token_jaccard", "title_token_recall",
    "title_token_precision", "query_bigram_recall", "query_number_recall", "negation_mismatch",
    "log_document_length", "log_query_length", "query_term_frequency",
)
DENSE_FEATURES = ("dense_score", "dense_reciprocal_rank", "rrf_score")
BASE_FEATURES = ("log_bm25_score", "bm25_reciprocal_rank")
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
NEGATIONS = frozenset(("no", "not", "without", "neither", "never", "absence"))
LOGGER = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def feature_names(include_dense: bool = True) -> list[str]:
    return list(BASE_FEATURES + (DENSE_FEATURES if include_dense else ()) + LEXICAL_FEATURES)


def extract_features(query: str, hit: dict, include_dense: bool = True) -> list[float]:
    """Features use only the query, document, and unsupervised retrieval scores."""
    q = _tokens(query)
    title = _tokens(str(hit.get("title", "")))
    d = title + _tokens(str(hit.get("text", "")))
    qs, ds, ts = set(q), set(d), set(title)
    overlap = len(qs & ds)
    bigrams_q, bigrams_d = set(zip(q, q[1:])), set(zip(d, d[1:]))
    numbers_q = {token for token in qs if any(char.isdigit() for char in token)}
    counts = Counter(d)
    bm25 = max(0.0, float(hit.get("bm25_score") or 0.0))
    bm25_rank = max(1, int(hit.get("bm25_rank") or 10**9))
    result = [math.log1p(bm25), 1.0 / bm25_rank if bm25 > 0 else 0.0]
    if include_dense:
        result += [float(hit.get("dense_score") or 0.0), 1.0 / max(1, int(hit.get("dense_rank") or 10**9)), float(hit.get("rrf_score") or 0.0)]
    result += [
        overlap / max(1, len(qs)), overlap / max(1, len(ds)), overlap / max(1, len(qs | ds)),
        len(qs & ts) / max(1, len(qs)), len(qs & ts) / max(1, len(ts)),
        len(bigrams_q & bigrams_d) / max(1, len(bigrams_q)),
        len(numbers_q & ds) / max(1, len(numbers_q)),
        float(bool(qs & NEGATIONS) != bool(ds & NEGATIONS)), math.log1p(len(d)), math.log1p(len(q)),
        sum(math.log1p(counts[token]) for token in qs) / max(1, len(qs)),
    ]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Nonfinite retrieval score in feature input")
    return result


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def normalized_claim(query: str) -> str:
    # Numeric variants share a family; shared gold documents capture most other variants.
    return " ".join(re.sub(r"\d+", "<num>", token) for token in _tokens(query))


def grouped_train_validation_split(
    queries: dict[str, str], qrels: dict[str, dict[str, int]], *, seed: int = SEED, validation_fraction: float = 0.2
) -> tuple[list[str], list[str], dict]:
    """Connected components prevent shared positive evidence/claim variants crossing splits."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    qids = sorted(qid for qid, rels in qrels.items() if any(int(rel) > 0 for rel in rels.values()))
    if len(qids) < 2:
        raise ValueError("At least two labeled queries are required")
    parent = {qid: qid for qid in qids}

    def find(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen_docs, seen_claims = {}, {}
    for qid in qids:
        signature = normalized_claim(queries[qid])
        if signature in seen_claims:
            union(qid, seen_claims[signature])
        seen_claims[signature] = qid
        for doc_id, rel in qrels[qid].items():
            if int(rel) <= 0:
                continue
            if doc_id in seen_docs:
                union(qid, seen_docs[doc_id])
            seen_docs[doc_id] = qid
    groups: dict[str, list[str]] = {}
    for qid in qids:
        groups.setdefault(find(qid), []).append(qid)
    roots = sorted(groups)
    if len(roots) < 2:
        raise ValueError("All claims are one evidence-connected group; a leakage-safe validation split is impossible")
    np.random.default_rng(seed).shuffle(roots)
    validation, train = [], []
    target = max(1, round(len(qids) * validation_fraction))
    for i, root in enumerate(roots):
        destination = validation if len(validation) < target and i < len(roots) - 1 else train
        destination.extend(groups[root])
    train, validation = sorted(train), sorted(validation)
    train_docs = {d for q in train for d, rel in qrels[q].items() if int(rel) > 0}
    val_docs = {d for q in validation for d, rel in qrels[q].items() if int(rel) > 0}
    train_claims = {normalized_claim(queries[q]) for q in train}
    val_claims = {normalized_claim(queries[q]) for q in validation}
    assert not train_docs & val_docs and not train_claims & val_claims
    return train, validation, {
        "seed": seed, "requested_validation_fraction": validation_fraction,
        "train_query_count": len(train), "validation_query_count": len(validation), "group_count": len(groups),
        "largest_group_size": max(map(len, groups.values())), "train_query_ids": train, "validation_query_ids": validation,
        "shared_positive_documents": 0, "shared_normalized_claims": 0,
        "group_definition": "connected components of shared positive documents or equal lowercase punctuation-free claims with digits normalized",
        "split_sha256": _hash({"train": train, "validation": validation}),
    }


class EvidenceRanker:
    """LightGBM text-model serialization with an explicit feature schema."""

    def __init__(self, *, include_dense: bool = True):
        self.include_dense = include_dense
        self.booster: lgb.Booster | None = None
        self.metadata: dict = {}

    @staticmethod
    def _training_candidates(dataset, index, qids: list[str]) -> tuple[dict[str, list[dict]], int]:
        candidates, injected = {}, 0
        for position, qid in enumerate(qids, 1):
            query = dataset.queries[qid]
            hits = index.search(query, method="rrf", k=POOL_SIZE)
            present = {str(hit["doc_id"]) for hit in hits}
            missing = {str(doc) for doc, rel in dataset.qrels["train"][qid].items() if int(rel) > 0} - present
            if missing:
                all_hits = index.search(query, method="rrf", k=len(dataset.docs))
                extra = [hit for hit in all_hits if str(hit["doc_id"]) in missing]
                if len(extra) != len(missing):
                    raise ValueError(f"Gold documents missing from the corpus for training query {qid}")
                hits = hits + extra
                injected += len(extra)
            candidates[qid] = hits
            if position % 100 == 0 or position == len(qids):
                LOGGER.info("Training candidates: %s/%s queries; %s injected positives", position, len(qids), injected)
        return candidates, injected

    def fit(self, dataset, index, *, seed: int = SEED, threads: int = 2, include_dense: bool | None = None) -> dict:
        import lightgbm as lgb
        if include_dense is not None:
            self.include_dense = include_dense
        self.metadata = {"threads": max(1, threads)}
        started = time.perf_counter()
        LOGGER.info("Training LambdaMART (%s features)", "full" if self.include_dense else "no dense/RRF")
        train_qids, val_qids, split = grouped_train_validation_split(dataset.queries, dataset.qrels["train"], seed=seed)
        all_qids = sorted(train_qids + val_qids)
        candidates, injected = self._training_candidates(dataset, index, all_qids)
        # Tuning evaluates natural retrieved pools: injected positives are NEVER exposed to validation ranking.
        val_candidates = {qid: index.search(dataset.queries[qid], method="rrf", k=POOL_SIZE) for qid in val_qids}

        def matrices(qids: list[str]):
            x, y, sizes = [], [], []
            for qid in qids:
                hits = candidates[qid]
                sizes.append(len(hits))
                x.extend(extract_features(dataset.queries[qid], hit, self.include_dense) for hit in hits)
                y.extend(max(0, int(dataset.qrels["train"][qid].get(str(hit["doc_id"]), 0))) for hit in hits)
            return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), sizes

        train_x, train_y, train_groups = matrices(train_qids)
        fixed = dict(BASE_CONFIG, objective="lambdarank", metric="ndcg", random_state=seed, n_jobs=max(1, threads), verbosity=-1,
                     deterministic=True, force_col_wise=True)
        trials, best_config, best_score = [], None, -math.inf
        for config in GRID:
            LOGGER.info("Validating LambdaMART config %s", dict(BASE_CONFIG, **config))
            model = lgb.LGBMRanker(**fixed, **config)
            model.fit(train_x, train_y, group=train_groups, feature_name=feature_names(self.include_dense))
            self.booster = model.booster_
            metrics = macro_metrics([query_metrics(self.rerank(dataset.queries[qid], val_candidates[qid]), dataset.qrels["train"][qid]) for qid in val_qids])
            trials.append({"config": dict(BASE_CONFIG, **config), "validation_metrics": metrics})
            LOGGER.info("Validation nDCG@10=%.6f MRR@10=%.6f", metrics["ndcg@10"], metrics["mrr@10"])
            if metrics["ndcg@10"] > best_score:
                best_config, best_score = dict(config), metrics["ndcg@10"]
        assert best_config is not None
        baseline_metrics = macro_metrics([query_metrics(val_candidates[qid], dataset.qrels["train"][qid]) for qid in val_qids])
        all_x, all_y, all_groups = matrices(all_qids)
        final_model = lgb.LGBMRanker(**fixed, **best_config)
        final_model.fit(all_x, all_y, group=all_groups, feature_name=feature_names(self.include_dense))
        self.booster = final_model.booster_
        gains = self.booster.feature_importance(importance_type="gain")
        self.metadata = {
            "schema_version": SCHEMA_VERSION, "include_dense": self.include_dense, "feature_names": feature_names(self.include_dense),
            "algorithm": "LightGBM LambdaMART", "lightgbm_version": lgb.__version__, "seed": seed, "threads": max(1, threads),
            "candidate_method": "rrf", "candidate_pool_size": POOL_SIZE,
            "candidate_ablation_note": "No-dense feature ablation uses identical hybrid candidates; this isolates feature effects, not dense retrieval effects.",
            "split": split, "hyperparameter_grid": [dict(BASE_CONFIG, **config) for config in GRID], "trials": trials,
            "selection_metric": "validation ndcg@10; stable first-grid-item tie break",
            "selected_config": dict(BASE_CONFIG, **best_config), "selected_validation_ndcg@10": best_score,
            "validation_rrf_metrics": baseline_metrics,
            "validation_recommendation": "learned" if best_score > baseline_metrics["ndcg@10"] else "rrf",
            "final_fit_query_count": len(all_qids), "final_fit_candidate_count": len(all_y),
            "positive_injection_count": injected,
            "tuning_train_positive_injection_count": sum(max(0, len(candidates[qid]) - POOL_SIZE) for qid in train_qids),
            "validation_positive_injection_count": 0, "test_positive_injection_count": 0,
            "positive_injection_policy": "Only supervised training matrices contain missing gold positives. Validation/test ranked pools are untouched. Final refit uses all official train queries.",
            "unjudged_policy": "Unjudged training candidates are assigned label zero; sparse judgments can create false negatives.",
            "training_queries_sha256": _hash({qid: dataset.queries[qid] for qid in all_qids}),
            "training_qrels_sha256": _hash(dataset.qrels["train"]),
            "corpus_sha256": getattr(dataset, "corpus_hash", None) or _hash(dataset.docs),
            "retrieval_metadata": dict(getattr(index, "metadata", {})),
            "feature_gain": {name: float(gain) for name, gain in zip(feature_names(self.include_dense), gains)},
            "elapsed_seconds": time.perf_counter() - started,
        }
        LOGGER.info("LambdaMART complete: validation nDCG@10 %.6f vs RRF %.6f; %.1f seconds",
                    best_score, baseline_metrics["ndcg@10"], self.metadata["elapsed_seconds"])
        return self.metadata

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if self.booster is None:
            raise RuntimeError("Ranker must be fitted or loaded before prediction")
        if not hits:
            return []
        matrix = np.asarray([extract_features(query, hit, self.include_dense) for hit in hits], dtype=np.float32)
        # Explicit thread count avoids OpenMP oversubscription on many-core hosts.
        scores = self.booster.predict(matrix, num_threads=max(1, int(self.metadata.get("threads", 2))))
        ranked = [dict(hit, ltr_score=float(score)) for hit, score in zip(hits, scores)]
        return sorted(ranked, key=lambda hit: (-hit["ltr_score"], str(hit["doc_id"])))

    def save(self, model_dir: Path) -> None:
        if self.booster is None or not self.metadata:
            raise RuntimeError("No fitted model to save")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_dir / "model.txt"))
        self.metadata["model_sha256"] = hashlib.sha256((model_dir / "model.txt").read_bytes()).hexdigest()
        (model_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def validate_index(self, metadata: dict) -> None:
        """Reject prediction against a different corpus, encoder, or scoring recipe."""
        expected = self.metadata.get("retrieval_metadata", {})
        identity_keys = ("corpus_sha256", "model_id", "model_revision", "model_sha256", "tokenizer_sha256", "max_length",
                         "backend", "text_format", "bm25_k1", "bm25_b", "rrf_k", "tokenizer", "dense_enabled",
                         "rrf_zero_bm25_policy", "pooling", "normalize", "dimension", "tie_break", "ranking_scope")
        if not expected.get("corpus_sha256"):
            raise ValueError("Model lacks training-index identity metadata")
        changed = [key for key in identity_keys if key in expected and expected[key] != metadata.get(key)]
        if changed:
            raise ValueError("Model and retrieval index are incompatible: " + ", ".join(changed))

    @classmethod
    def load(cls, model_dir: Path) -> "EvidenceRanker":
        import lightgbm as lgb
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(include_dense=bool(metadata["include_dense"]))
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("feature_names") != feature_names(instance.include_dense):
            raise ValueError("Saved model feature schema is incompatible")
        if metadata.get("model_sha256") != hashlib.sha256((model_dir / "model.txt").read_bytes()).hexdigest():
            raise ValueError("Saved model checksum does not match metadata")
        instance.metadata = metadata
        instance.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        if instance.booster.feature_name() != feature_names(instance.include_dense):
            raise ValueError("Model and metadata feature names do not match")
        return instance
