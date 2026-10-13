"""Sparse BM25, exact MiniLM retrieval and reciprocal-rank fusion."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from scipy import sparse

from .data import Dataset
from .neural import MiniLMEncoder

LOGGER = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text.casefold())


def embedding_cache_key(corpus_hash: str, encoder_metadata: dict, text_format: str = "title + newline + text") -> str:
    return hashlib.sha256(json.dumps({
        "corpus": corpus_hash,
        "encoder": {key: value for key, value in encoder_metadata.items() if key != "threads"},
        "text_format": text_format,
    }, sort_keys=True).encode()).hexdigest()


class RetrievalIndex:
    """Scores the entire corpus; never consults relevance labels during search.

    BM25 search skips query encoding and reports dense components as None.
    Dense and RRF search include both component scores/ranks. Query embeddings
    are not cached, so measured latency includes encoding on every call.
    """

    def __init__(self, dataset: Dataset, cache_dir: Path, dense: bool = True, threads: int = 2, encoder=None):
        self.dataset = dataset
        self.docs = sorted(dataset.docs, key=lambda doc: str(doc["_id"]))
        self.cache_dir = Path(cache_dir)
        self.dense_enabled = dense
        self.threads = threads
        self.k1, self.b, self.rrf_k = 1.2, 0.75, 60
        self.metadata = {
            "corpus_sha256": dataset.corpus_hash, "corpus_hash": dataset.corpus_hash,
            "documents": len(self.docs), "dense_enabled": dense,
            "bm25_k1": self.k1, "bm25_b": self.b, "rrf_k": self.rrf_k,
            "tokenizer": "unicode-word-casefold-v1", "text_format": "title + newline + text",
            "tie_break": "document ID lexicographic ascending", "ranking_scope": "full corpus",
            "query_embedding_cache": False,
            "rrf_zero_bm25_policy": "exclude lexical vote for nonmatching documents",
        }
        self.encoder = None
        self.encoder_override = encoder
        self.embeddings = None
        self._built = False

    def build(self):
        if not self.docs:
            raise ValueError("Cannot index an empty corpus")
        if len({str(doc["_id"]) for doc in self.docs}) != len(self.docs):
            raise ValueError("Duplicate document IDs")
        texts = [f"{doc.get('title', '')}\n{doc['text']}" for doc in self.docs]
        counters = [Counter(tokenize(text)) for text in texts]
        self.vocabulary = {term: i for i, term in enumerate(sorted({term for row in counters for term in row}))}
        lengths = np.asarray([sum(row.values()) for row in counters], dtype=np.float64)
        avg_length = max(float(lengths.mean()), 1.0)
        rows, columns, values = [], [], []
        for row_id, counter in enumerate(counters):
            for term, frequency in counter.items():
                rows.append(row_id)
                columns.append(self.vocabulary[term])
                values.append(frequency)
        counts = sparse.csr_matrix((values, (rows, columns)), shape=(len(self.docs), len(self.vocabulary)), dtype=np.float64)
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(self.docs) - df + 0.5) / (df + 0.5))
        coo = counts.tocoo()
        denominator = coo.data + self.k1 * (1.0 - self.b + self.b * lengths[coo.row] / avg_length)
        weights = idf[coo.col] * coo.data * (self.k1 + 1.0) / denominator
        self.bm25 = sparse.csc_matrix((weights, (coo.row, coo.col)), shape=counts.shape)
        self.metadata["vocabulary_size"] = len(self.vocabulary)
        if self.dense_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.encoder = self.encoder_override or MiniLMEncoder(self.cache_dir, threads=self.threads)
            self.metadata.update(self.encoder.metadata)
            fingerprint = embedding_cache_key(self.dataset.corpus_hash, self.encoder.metadata, self.metadata["text_format"])
            cache_path = self.cache_dir / f"embeddings-{fingerprint}.npy"
            if cache_path.exists():
                vectors = np.load(cache_path, allow_pickle=False)
                expected = (len(self.docs), self.encoder.metadata["dimension"])
                if vectors.shape != expected or not np.isfinite(vectors).all():
                    raise ValueError(f"Invalid embedding cache: {cache_path}; delete it and rebuild")
                if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
                    raise ValueError(f"Embedding cache is not L2-normalized: {cache_path}; delete it and rebuild")
                self.embeddings = vectors.astype(np.float32, copy=False)
                LOGGER.info("Loaded embedding cache for %s documents", len(self.docs))
            else:
                self.embeddings = self.encoder.encode(texts)
                fd, temp_name = tempfile.mkstemp(prefix=".embeddings-", dir=self.cache_dir)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        np.save(handle, self.embeddings, allow_pickle=False)
                    os.replace(temp_name, cache_path)
                except BaseException:
                    Path(temp_name).unlink(missing_ok=True)
                    raise
            self.metadata["embedding_cache_key"] = fingerprint
        self._built = True
        return self

    @staticmethod
    def _ranks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Documents are already sorted by ID; stable sort preserves tie ordering.
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        return order, ranks

    def search(self, query: str, method: str = "rrf", k: int = 50) -> list[dict]:
        if not self._built:
            raise RuntimeError("Call build() before search()")
        if method not in {"bm25", "dense", "rrf"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        if method != "bm25" and not self.dense_enabled:
            raise ValueError(f"{method} requires dense=True; no silent lexical fallback is allowed")
        if not isinstance(query, str) or not query.strip() or not tokenize(query):
            raise ValueError("Query must contain at least one word or number")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        started = time.perf_counter()
        terms = Counter(tokenize(query))
        columns = [self.vocabulary[term] for term in terms if term in self.vocabulary]
        frequencies = np.asarray([terms[term] for term in terms if term in self.vocabulary], dtype=np.float64)
        bm25_scores = np.asarray(self.bm25[:, columns] @ frequencies).ravel() if columns else np.zeros(len(self.docs))
        bm25_order, bm25_ranks = self._ranks(bm25_scores)
        dense_scores = dense_ranks = rrf_scores = None
        order = bm25_order[bm25_scores[bm25_order] > 0]
        if self.dense_enabled and method != "bm25":
            query_vector = self.encoder.encode([query])[0]
            dense_scores = self.embeddings @ query_vector
            dense_order, dense_ranks = self._ranks(dense_scores)
            # Nonmatching lexical documents have score zero. Their deterministic
            # ID tie order is useful for stable features, but is not evidence of
            # lexical relevance and must not add an arbitrary fusion bonus.
            lexical_votes = np.where(bm25_scores > 0, 1.0 / (self.rrf_k + bm25_ranks), 0.0)
            rrf_scores = lexical_votes + 1.0 / (self.rrf_k + dense_ranks)
            if method == "dense":
                order = dense_order
            elif method == "rrf":
                order, _ = self._ranks(rrf_scores)
        self.last_search_ms = (time.perf_counter() - started) * 1000
        hits = []
        for position in order[:min(k, len(self.docs))]:
            doc = self.docs[position]
            hits.append({
                "doc_id": str(doc["_id"]), "title": doc.get("title", ""), "text": doc["text"],
                "bm25_score": float(bm25_scores[position]), "bm25_rank": int(bm25_ranks[position]),
                "dense_score": float(dense_scores[position]) if dense_scores is not None else None,
                "dense_rank": int(dense_ranks[position]) if dense_ranks is not None else None,
                "rrf_score": float(rrf_scores[position]) if rrf_scores is not None else None,
            })
        return hits
