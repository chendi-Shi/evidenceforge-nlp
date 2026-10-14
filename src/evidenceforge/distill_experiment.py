"""Budgeted local retrieval distillation; source validation precedes held-out evaluation.

Teacher inference, student updates, and evaluation are separate resumable phases.
No hosted inference, cloud accelerator, or paid API is used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numpy as np

from .modern import QwenEmbeddingEncoder, _memory_snapshot

LOGGER = logging.getLogger(__name__)
WEB_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
ARMS = {"ce": (0.0, 0.0), "kd": (1.0, 0.0), "anchor": (0.0, 10.0), "kd_anchor": (1.0, 10.0)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_rank_files(directory, result):
    for filename_key, hash_key in (("run_filename", "run_sha256"), ("per_query_filename", "per_query_sha256")):
        filename = result[filename_key]
        if Path(filename).name != filename or file_digest(Path(directory) / filename) != result[hash_key]:
            raise ValueError("Saved ranking or per-query artifact is missing or changed")


@dataclass(frozen=True)
class ExperimentConfig:
    train_queries: int = 12000
    validation_queries: int = 1000
    data_seed: int = 42
    seeds: tuple[int, ...] = (17, 42, 73)
    student_tokens: int = 256
    teacher_tokens: int = 512
    suffix_layers: int = 2
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 2e-5
    temperature: float = 0.05
    validation_every: int = 250
    threads: int = 2

    def __post_init__(self):
        if self.epochs != 1:
            raise ValueError("This preregistered distillation runner supports exactly one epoch")
        if self.train_queries < 2 or self.validation_queries < 1 or self.batch_size < 2:
            raise ValueError("Invalid source size or training batch size")
        if self.train_queries % self.batch_size == 1:
            raise ValueError("A singleton final batch is not permitted")
        if self.validation_every < 1:
            raise ValueError("Validation period must be positive")


class BudgetExpired(RuntimeError):
    """A completed atomic batch remains resumable after a local time limit."""


class Progress:
    def __init__(self, output_dir: Path, max_hours: float):
        if not 0 < max_hours <= 12:
            raise ValueError("A local invocation must have a positive budget of at most 12 hours")
        self.path = Path(output_dir) / "live_status.json"
        self.started = time.monotonic()
        self.deadline = self.started + max_hours * 3600
        self.phase = "starting"

    def update(self, phase, **fields):
        self.phase = phase
        write_json(self.path, {"phase": phase, "pid": os.getpid(),
                              "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                              "invocation_elapsed_seconds": time.monotonic() - self.started, **fields})
        if time.monotonic() >= self.deadline:
            raise BudgetExpired("Local invocation time limit reached; resume preserves completed batches")


@contextmanager
def single_pipeline():
    """OS-owned mutex releases on process death; no stale lock-file guessing."""
    if os.name == "nt":
        import ctypes
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\EvidenceForgeDistillationV3")
        if not mutex or ctypes.windll.kernel32.GetLastError() == 183:
            if mutex:
                ctypes.windll.kernel32.CloseHandle(mutex)
            raise RuntimeError("An EvidenceForge distillation invocation is already active")
        try:
            yield
        finally:
            ctypes.windll.kernel32.CloseHandle(mutex)
    else:
        # The resource-bounded overnight runner currently targets this Windows host.
        # Individual data/model/evaluation functions remain portable.
        raise RuntimeError("The unattended runner currently requires a Windows process mutex")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def row_text(row, role):
    # Explicit text aliases permit independent source preparation to retain clear column names.
    if role == "query":
        return row["query"]
    return row.get(role + "_text", row.get(role))


def source_items(rows):
    queries, documents = {}, {}
    for row in rows:
        for target, key, text in [(queries, row["query_id"], row["query"]),
                                  (documents, row["positive_id"], row_text(row, "positive")),
                                  (documents, row["negative_id"], row_text(row, "negative"))]:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Source text must be a nonempty string")
            if key in target and target[key] != text:
                raise ValueError("One source ID maps to different texts")
            target[key] = text
    return dict(sorted(queries.items())), dict(sorted(documents.items()))


def prepare(data_dir, output_dir, config):
    from .distill_data import prepare_msmarco_distill
    manifest = prepare_msmarco_distill(Path(data_dir), Path(data_dir) / "prepared",
                                      n_train=config.train_queries, n_validation=config.validation_queries,
                                      seed=config.data_seed)
    write_json(Path(output_dir) / "data_manifest.json", manifest)
    return manifest


def source_paths(data_dir):
    directory = Path(data_dir) / "prepared"
    return directory / "train.jsonl", directory / "validation.jsonl"


def protocol(output_dir, data_dir, config, base_model_dir):
    from importlib.metadata import version
    train_path, validation_path = source_paths(data_dir)
    teacher = QwenEmbeddingEncoder(Path("cache"), threads=config.threads,
                                   max_tokens=config.teacher_tokens, instruction=WEB_INSTRUCTION, backend="vulkan")
    spec = {"version": 3, "config": asdict(config), "arms": ARMS,
            "train_sha256": file_digest(train_path), "validation_sha256": file_digest(validation_path),
            "base_model_sha256": file_digest(Path(base_model_dir) / "model.safetensors"),
            "base_config_sha256": file_digest(Path(base_model_dir) / "config.json"),
            "base_tokenizer_sha256": file_digest(Path(base_model_dir) / "tokenizer.json"),
            "student_configs": {arm: asdict(student_config(config, arm=arm)) for arm in ARMS},
            "implementation_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in
                                      ("distill_experiment.py", "distill_student.py", "distill_data.py", "distill_eval.py")},
            "dependencies": {name: version(name) for name in ("torch", "transformers", "numpy", "safetensors", "pyarrow")},
            "teacher": teacher.metadata,
            "training": "Freeze embeddings and first four MiniLM blocks; update all parameters in last two blocks",
            "dropout": "Disabled during cached-prefix training and inference; optimizer gradients remain enabled",
            "objective": "CE + alpha KL(softmax(teacher cosine/.05)||softmax(student cosine/.05)) + beta base cosine distance; no T squared",
            "candidate_pool": "Shared batch of B positives and B source BM25 negatives; duplicates/known-positive negatives masked",
            "validation": "All validation queries rank the same union of validation positive/negative documents; sampled corpus, not official full MS MARCO",
            "checkpoint_selection": "Maximum source-validation nDCG@10 at fixed 250-step checkpoints and final step, earliest step wins ties; step zero is reported but not selectable",
            "primary": "SCIDOCS kd_anchor minus kd nDCG@10, per seed; separately require comparison to frozen base and source validation to interpret retention",
            "secondary": "All four arms, interaction contrast, frozen base, query/document drift and source validation",
            "held_out": "SCIDOCS official test with full corpus, opened only after all selected weights are frozen; no target adaptation or selection",
            "regression_only": ["SciFact", "NFCorpus"],
            "uncertainty": "Three separate optimizer seeds, fixed source split; no pooling seed-query cells as independent samples",
            "claims_not_made": ["novel algorithm", "SOTA", "LLM instruction tuning", "full MS MARCO evaluation", "guaranteed improvement"]}
    spec = json.loads(json.dumps(spec))
    path = Path(output_dir) / "protocol.json"
    value = {"spec": spec, "spec_sha256": digest(spec)}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["spec_sha256"] != value["spec_sha256"] or previous["spec"] != spec:
            raise ValueError("Existing experiment protocol differs; use a new output directory")
        return previous
    value["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, value)
    return value


def teacher_vectors(rows, cache_dir, work_dir, output_dir, config, progress, *, pilot=False):
    queries, documents = source_items(rows)
    teacher = QwenEmbeddingEncoder(Path(cache_dir), threads=config.threads,
                                   max_tokens=config.teacher_tokens, instruction=WEB_INSTRUCTION, backend="vulkan")
    report = {"purpose": "throughput pilot, no retrieval claim" if pilot else "frozen offline distillation targets",
              "teacher": teacher.metadata, "unique_queries": len(queries), "unique_documents": len(documents),
              "memory_before": _memory_snapshot(), "phases": {}}
    destination = Path(work_dir) / ("pilot_teacher.npz" if pilot else "teacher.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_identity = digest({"query_texts": queries, "document_texts": documents, "teacher": teacher.metadata})
    manifest_path = destination.with_suffix(".json")
    if destination.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != expected_identity or manifest["sha256"] != file_digest(destination):
            raise ValueError("Teacher vector artifact identity mismatch")
        return destination, manifest["report"]
    arrays = {}
    try:
        for role, items in (("queries", queries), ("documents", documents)):
            ids, texts = list(items), list(items.values())
            vectors, timings = [], []
            stats = {"items": 0, "cache_hits": 0, "truncated": 0, "tokens": 0, "fresh_inference_seconds": 0.0}
            started = time.perf_counter()
            for start in range(0, len(ids), 16):
                progress.update("pilot_teacher" if pilot else "teacher", role=role, completed=start, total=len(ids))
                result = teacher.encode(texts[start:start + 16], is_query=role == "queries",
                                        use_cache=not pilot, cache_namespace="distillation_v3")
                vectors.append(result)
                batch_stats = teacher.last_stats
                for key in ("items", "cache_hits", "truncated", "tokens"):
                    stats[key] += batch_stats[key]
                timings.extend(batch_stats["uncached_item_seconds"])
                stats["fresh_inference_seconds"] += sum(batch_stats["uncached_item_seconds"])
                write_json(Path(output_dir) / f"{'pilot_' if pilot else ''}teacher_{role}_progress.json", stats)
            stats.update(wall_seconds=time.perf_counter() - started,
                         fresh_latency_p50_seconds=float(np.median(timings)) if timings else None,
                         fresh_latency_p95_seconds=float(np.quantile(timings, .95)) if timings else None)
            report["phases"][role] = stats
            arrays[role + "_ids"] = np.asarray(ids)
            arrays[role + "_vectors"] = np.concatenate(vectors)
        report["memory_at_end"] = _memory_snapshot(teacher.process)
    finally:
        teacher.close()
    with destination.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(destination.with_suffix(".tmp"), destination)
    manifest = {"identity": expected_identity, "sha256": file_digest(destination), "report": report}
    write_json(manifest_path, manifest)
    write_json(Path(output_dir) / ("pilot_teacher.json" if pilot else "teacher_report.json"), manifest)
    return destination, report


def student_config(config, seed=42, arm="ce"):
    from .distill_student import DistillConfig
    alpha, beta = ARMS[arm]
    return DistillConfig(suffix_layers=config.suffix_layers, max_length=config.student_tokens,
                         threads=config.threads, seed=seed, learning_rate=config.learning_rate,
                         ce_temperature=config.temperature, kd_temperature=config.temperature,
                         kd_weight=alpha, anchor_weight=beta)


def encode_student(student, items, progress, phase, *, base=False):
    ids, texts = list(items), list(items.values())
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(ids), 16):
        progress.update(phase, completed=start, total=len(ids))
        batch_ids, batch_texts = ids[start:start + 16], texts[start:start + 16]
        if base:
            vector = student.base_vectors(batch_texts, batch_ids)
        else:
            vector = student.encode(batch_texts, prefix_ids=batch_ids, batch_size=16)
        vectors.append(vector)
    return np.concatenate(vectors), {"items": len(ids), "wall_seconds": time.perf_counter() - started,
                                     "includes_cached_prefix_reads": True, "end_to_end_latency": False}


def prepare_prefixes(rows, base_model_dir, work_dir, output_dir, config, progress):
    from .distill_student import DistillStudent
    from tokenizers import Tokenizer
    queries, documents = source_items(rows)
    items = {"q-" + key: text for key, text in queries.items()}
    items.update({"d-" + key: text for key, text in documents.items()})
    student = DistillStudent(Path(base_model_dir), student_config(config), prefix_cache_dir=Path(work_dir) / "prefix")
    ids, texts = list(items), list(items.values())
    base_path = Path(work_dir) / "source_base_vectors.npz"
    base_manifest_path = base_path.with_suffix(".json")
    base_identity = digest({"items": items, "student": student.identity})
    if base_path.exists() and base_manifest_path.exists():
        saved = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        if saved["identity"] != base_identity or saved["sha256"] != file_digest(base_path):
            raise ValueError("Frozen source-base vector identity mismatch")
        return saved["report"]
    started = time.perf_counter()
    original_vectors = []
    length_tokenizer = Tokenizer.from_file(str(Path(base_model_dir) / "tokenizer.json"))
    length_tokenizer.no_padding()
    length_tokenizer.no_truncation()
    lengths = {role: {"items": 0, "truncated": 0, "original_tokens": 0, "retained_tokens": 0}
               for role in ("queries", "documents")}
    for start in range(0, len(ids), 32):
        progress.update("source_prefix_cache", completed=start, total=len(ids))
        student.cache_prefixes(texts[start:start + 32], ids[start:start + 32], batch_size=8)
        original_vectors.append(student.base_vectors(texts[start:start + 32], ids[start:start + 32]))
        for key, encoding in zip(ids[start:start + 32], length_tokenizer.encode_batch(texts[start:start + 32]), strict=True):
            stats = lengths["queries" if key.startswith("q-") else "documents"]
            length = len(encoding.ids)
            stats["items"] += 1
            stats["truncated"] += int(length > config.student_tokens)
            stats["original_tokens"] += length
            stats["retained_tokens"] += min(length, config.student_tokens)
    check = student.verify_cached_equivalence(texts[:8], ids[:8])
    report = {"texts": len(ids), "seconds_this_invocation": time.perf_counter() - started,
              "trainable_parameters": student.trainable_parameter_count, "student": student.identity,
              "equivalence": check,
              "token_coverage": lengths,
              "note": "FP32 frozen-prefix cache and immutable original-base vectors; excludes teacher inference"}
    write_json(Path(output_dir) / "prefix_report.json", report)
    with base_path.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, ids=np.asarray(ids), vectors=np.concatenate(original_vectors))
    os.replace(base_path.with_suffix(".tmp"), base_path)
    write_json(base_manifest_path, {"identity": base_identity, "sha256": file_digest(base_path), "report": report})
    return report


def validation_items(rows):
    queries, documents = source_items(rows)
    qrels = {row["query_id"]: {doc_id: 1 for doc_id in row["known_positive_ids"] if doc_id in documents}
             for row in rows}
    return queries, documents, qrels


def validate_student(student, rows, output_path, progress, *, base=False, base_lookup=None):
    from .distill_eval import rank_embeddings
    queries, documents, qrels = validation_items(rows)
    q_items = {"q-" + key: text for key, text in queries.items()}
    d_items = {"d-" + key: text for key, text in documents.items()}
    q_base = (np.stack([base_lookup[key] for key in q_items]) if base_lookup is not None else
              student.base_vectors(list(queries.values()), list(q_items)))
    d_base = (np.stack([base_lookup[key] for key in d_items]) if base_lookup is not None else
              student.base_vectors(list(documents.values()), list(d_items)))
    if base:
        q_vectors, d_vectors = q_base, d_base
        q_time, d_time = {"precomputed_base": True}, {"precomputed_base": True}
    else:
        q_vectors, q_time = encode_student(student, q_items, progress, "source_validation_queries")
        d_vectors, d_time = encode_student(student, d_items, progress, "source_validation_documents")
    result = rank_embeddings(list(queries), q_vectors, list(documents), d_vectors, qrels,
                             Path(output_path), "source-validation", ignore_identical_ids=False)
    result["drift"] = {"query_mean_cosine_distance_to_base": float(np.mean(1 - (q_base * q_vectors).sum(1))),
                       "document_mean_cosine_distance_to_base": float(np.mean(1 - (d_base * d_vectors).sum(1)))}
    result["timing"] = {"queries": q_time, "documents": d_time}
    result["scope"] = "Fixed shared validation-document pool; not the 8.8M-passage official MS MARCO evaluation"
    return result


def training_targets(batch, teacher_queries, teacher_documents, query_locations, document_locations, student, base_lookup=None):
    queries = [row["query"] for row in batch]
    qids = [row["query_id"] for row in batch]
    ids = [row["positive_id"] for row in batch] + [row["negative_id"] for row in batch]
    texts = [row_text(row, "positive") for row in batch] + [row_text(row, "negative") for row in batch]
    positive = np.asarray([[item in set(row["known_positive_ids"]) for item in ids] for row in batch], dtype=bool)
    q_cache_ids, d_cache_ids = ["q-" + key for key in qids], ["d-" + key for key in ids]
    return queries, texts, positive, {
        "query_ids": q_cache_ids, "candidate_ids": d_cache_ids,
        "teacher_query_vectors": teacher_queries[[query_locations[key] for key in qids]],
        "teacher_candidate_vectors": teacher_documents[[document_locations[key] for key in ids]],
        "base_query_vectors": (np.stack([base_lookup[key] for key in q_cache_ids]) if base_lookup is not None
                               else student.base_vectors(queries, q_cache_ids)),
        "base_candidate_vectors": (np.stack([base_lookup[key] for key in d_cache_ids]) if base_lookup is not None
                                   else student.base_vectors(texts, d_cache_ids)),
    }


def train_all(data_dir, base_model_dir, work_dir, output_dir, config, progress, frozen_protocol, teacher_path):
    from .distill_student import DistillStudent, DistillTrainer
    train_path, validation_path = source_paths(data_dir)
    rows, validation = read_rows(train_path), read_rows(validation_path)
    with np.load(teacher_path, allow_pickle=False) as arrays:
        teacher_queries, teacher_documents = arrays["queries_vectors"], arrays["documents_vectors"]
        query_locations = {key: index for index, key in enumerate(arrays["queries_ids"].tolist())}
        document_locations = {key: index for index, key in enumerate(arrays["documents_ids"].tolist())}
    teacher_hash = file_digest(teacher_path)
    base_lookup = None
    base_path = Path(work_dir) / "source_base_vectors.npz"
    if base_path.exists():
        base_manifest = json.loads(base_path.with_suffix(".json").read_text(encoding="utf-8"))
        if file_digest(base_path) != base_manifest["sha256"]:
            raise ValueError("Source-base vectors changed before training")
        with np.load(base_path, allow_pickle=False) as arrays:
            vectors = arrays["vectors"]
            base_lookup = dict(zip(arrays["ids"].tolist(), vectors, strict=True))
    baseline_path = Path(output_dir) / "source_validation_base.json"
    if baseline_path.exists():
        previous = json.loads(baseline_path.read_text(encoding="utf-8"))
        if previous.get("spec_sha256") != frozen_protocol["spec_sha256"]:
            raise ValueError("Source baseline validation identity mismatch")
        verify_rank_files(Path(output_dir) / "validation", previous)
    else:
        base = DistillStudent(Path(base_model_dir), student_config(config), prefix_cache_dir=Path(work_dir) / "prefix")
        result = validate_student(base, validation, Path(output_dir) / "validation/base.trec", progress, base=True, base_lookup=base_lookup)
        result["spec_sha256"] = frozen_protocol["spec_sha256"]
        write_json(baseline_path, result)
        del base
    reports = {}
    for seed in config.seeds:
        order = np.random.default_rng(seed).permutation(len(rows)).tolist()
        ordered = [rows[index] for index in order]
        for arm in ARMS:
            run_dir = Path(output_dir) / "training" / arm / str(seed)
            report_path = run_dir / "report.json"
            plan = {"spec_sha256": frozen_protocol["spec_sha256"], "teacher_sha256": teacher_hash,
                    "seed": seed, "arm": arm, "order_sha256": digest([row["query_id"] for row in ordered])}
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report["plan"] != plan:
                    raise ValueError("Completed training run identity differs")
                verify_training_run(run_dir, report, config)
                reports[f"{arm}/{seed}"] = report
                continue
            student = DistillStudent(Path(base_model_dir), student_config(config, seed, arm),
                                     prefix_cache_dir=Path(work_dir) / "prefix")
            trainer = DistillTrainer(student, plan, [row["query_id"] for row in ordered])
            checkpoint_dir = Path(work_dir) / "checkpoints" / arm / str(seed)
            trace_path = run_dir / "trace.json"
            state_path = run_dir / "selection.json"
            if (checkpoint_dir / "checkpoint.json").exists():
                trainer.load_checkpoint(checkpoint_dir)
            trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
            trace = [record for record in trace if record["step"] <= trainer.progress["completed_steps"]]
            if [record["step"] for record in trace] != list(range(1, trainer.progress["completed_steps"] + 1)):
                raise ValueError("Checkpoint gradient trace is missing, duplicated, or out of order")
            selection = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
                "best_step": None, "best_ndcg": None, "validations": []}
            # A checkpoint is saved before validation and best weights are selected.
            # After interruption repeat a missing validation of that exact checkpoint.
            started = time.perf_counter()

            def do_validation():
                step = trainer.progress["completed_steps"]
                if any(record["step"] == step for record in selection["validations"]):
                    return
                evaluation = validate_student(student, validation, run_dir / "validation" / f"step-{step}.trec", progress, base_lookup=base_lookup)
                score = evaluation["metrics"]["ndcg@10"]
                selection["validations"].append({"step": step, "metrics": evaluation["metrics"], "drift": evaluation["drift"]})
                if selection["best_ndcg"] is None or score > selection["best_ndcg"]:
                    student.save_suffix(run_dir / "selected")
                    selection.update(best_step=step, best_ndcg=score)
                write_json(state_path, selection)

            cursor = trainer.progress["row_cursor"]
            if trainer.progress["completed_steps"] and (trainer.progress["completed_steps"] % config.validation_every == 0 or cursor == len(ordered)):
                do_validation()
            try:
                for start in range(cursor, len(ordered), config.batch_size):
                    progress.update("training", arm=arm, seed=seed, step=trainer.progress["completed_steps"],
                                    examples_completed=start, examples_total=len(ordered))
                    batch = ordered[start:start + config.batch_size]
                    queries, texts, positive, targets = training_targets(batch, teacher_queries, teacher_documents,
                                                                         query_locations, document_locations, student, base_lookup)
                    record = trainer.step(queries, texts, positive, next_row_cursor=start + len(batch), **targets)
                    trace.append(record)
                    step = trainer.progress["completed_steps"]
                    if step % 50 == 0 or step % config.validation_every == 0 or start + len(batch) == len(ordered):
                        write_json(trace_path, trace)
                        trainer.save_checkpoint(checkpoint_dir)
                    if step % config.validation_every == 0 or start + len(batch) == len(ordered):
                        do_validation()
            except BudgetExpired:
                write_json(trace_path, trace)
                trainer.save_checkpoint(checkpoint_dir)
                raise
            final_metadata = student.save_suffix(run_dir / "final")
            report = {"plan": plan, "progress": trainer.progress, "trainable_parameters": student.trainable_parameter_count,
                      "selection": selection, "final_suffix": final_metadata,
                      "current_invocation_training_seconds": time.perf_counter() - started,
                      "timing_note": "Includes validation in this invocation only; resumed earlier time is not reconstructed",
                      "selected_metadata_sha256": file_digest(run_dir / "selected/metadata.json"),
                      "source_validation": "Fixed shared pool; step zero not selectable; no target labels used"}
            write_json(report_path, report)
            verify_training_run(run_dir, report, config)
            reports[f"{arm}/{seed}"] = report
            trainer, student = None, None
    frozen = {"spec_sha256": frozen_protocol["spec_sha256"], "runs": reports,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    frozen_path = Path(output_dir) / "frozen_students.json"
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text(encoding="utf-8"))
        if previous["spec_sha256"] != frozen["spec_sha256"] or previous["runs"] != frozen["runs"]:
            raise ValueError("Frozen student manifest changed")
        return previous
    write_json(frozen_path, frozen)
    return frozen


def verify_training_run(run_dir, report, config):
    expected_steps = int(np.ceil(config.train_queries / config.batch_size))
    if (report["progress"]["completed_steps"] != expected_steps or
            report["progress"]["row_cursor"] != config.train_queries or
            report["progress"]["nonzero_gradient_steps"] != expected_steps):
        raise ValueError("Training completion counters do not match the protocol")
    trace = json.loads((Path(run_dir) / "trace.json").read_text(encoding="utf-8"))
    if [record["step"] for record in trace] != list(range(1, expected_steps + 1)):
        raise ValueError("Completed training trace is incomplete")
    metadata_path = Path(run_dir) / "selected/metadata.json"
    if file_digest(metadata_path) != report["selected_metadata_sha256"]:
        raise ValueError("Selected checkpoint metadata changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(metadata["filename"]).name != metadata["filename"]:
        raise ValueError("Unsafe selected checkpoint filename")
    if file_digest(metadata_path.parent / metadata["filename"]) != metadata["suffix_sha256"]:
        raise ValueError("Selected checkpoint weights changed")


def evaluate_students(data_dir, base_model_dir, work_dir, output_dir, config, progress, frozen):
    from .distill_eval import load_or_download_scidocs, rank_embeddings, summarize_seeds
    from .distill_student import DistillStudent
    from .distill_data import query_id
    expected_runs = {f"{arm}/{seed}" for seed in config.seeds for arm in ARMS}
    if set(frozen["runs"]) != expected_runs:
        raise ValueError("Cannot open held-out data before every configured student is frozen")
    protocol_path = Path(output_dir) / "protocol.json"
    if not protocol_path.exists() or json.loads(protocol_path.read_text(encoding="utf-8"))["spec_sha256"] != frozen["spec_sha256"]:
        raise ValueError("Frozen students and declared experiment protocol differ")
    # Validate every frozen selected manifest before the first target-data access.
    for key, report in frozen["runs"].items():
        metadata_path = Path(output_dir) / "training" / key / "selected/metadata.json"
        if file_digest(metadata_path) != report["selected_metadata_sha256"]:
            raise ValueError("Selected model changed after the freeze")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_digest(metadata_path.parent / metadata["filename"]) != metadata["suffix_sha256"]:
            raise ValueError("Frozen suffix weights changed")
    progress.update("held_out_data", dataset="BEIR/SCIDOCS", model_count=len(frozen["runs"]))
    dataset = load_or_download_scidocs(Path(data_dir).parent / "scidocs")
    queries = {key: dataset.queries[key] for key in sorted(dataset.qrels["test"])}
    documents = {doc["_id"]: doc["title"] + "\n" + doc["text"] for doc in dataset.docs}
    q_items = {"scidocs-q-" + key: text for key, text in queries.items()}
    d_items = {"scidocs-d-" + key: text for key, text in documents.items()}
    # Text-based overlap is an audit after model freeze, never a way to choose an arm.
    source_query_signatures = {row["query_id"] for row in read_rows(source_paths(data_dir)[0])}
    overlaps = [key for key, text in queries.items()
                if query_id(text) in source_query_signatures]
    runs = {}
    identities = [("base", "frozen")] + [(arm, str(seed)) for seed in config.seeds for arm in ARMS]
    for arm, seed in identities:
        out = Path(output_dir) / "scidocs" / arm / str(seed)
        result_path = out / "report.json"
        expected = {"spec_sha256": frozen["spec_sha256"], "dataset": dataset.corpus_hash,
                    "queries_sha256": digest(queries), "qrels_sha256": digest(dataset.qrels["test"]),
                    "arm": arm, "seed": seed,
                    "weights": "base" if arm == "base" else frozen["runs"][f"{arm}/{seed}"]["selected_metadata_sha256"]}
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["identity"] != expected:
                raise ValueError("Held-out report identity changed")
            verify_rank_files(out, result)
        else:
            student = DistillStudent(Path(base_model_dir), student_config(config, 42 if arm == "base" else int(seed),
                                                                         "ce" if arm == "base" else arm),
                                     prefix_cache_dir=Path(work_dir) / "prefix")
            if arm != "base":
                student.load_suffix(Path(output_dir) / "training" / arm / seed / "selected")
            q_vectors, q_timing = encode_student(student, q_items, progress, f"scidocs_{arm}_{seed}_queries", base=arm == "base")
            d_vectors, d_timing = encode_student(student, d_items, progress, f"scidocs_{arm}_{seed}_documents", base=arm == "base")
            result = rank_embeddings(list(queries), q_vectors, list(documents), d_vectors, dataset.qrels["test"],
                                     out / "ranking.trec", f"{arm}-{seed}", ignore_identical_ids=True)
            result.update(identity=expected, timing={"queries": q_timing, "documents": d_timing})
            write_json(result_path, result)
            del student
        runs.setdefault(arm, {})[seed] = result
    report = {"spec_sha256": frozen["spec_sha256"], "dataset": dataset.metadata,
              "training_query_text_overlap_ids": overlaps,
              "summary": summarize_seeds(runs),
              "runs": runs,
              "status": "student evaluation complete; no Qwen target-corpus score implied",
              "limitations": ["Fixed MS MARCO subset, not full MS MARCO training", "Classic BEIR SCIDOCS citation-retrieval task",
                              "Three optimizer seeds; no pretraining contamination audit",
                              "Query-level intervals per seed; no multiple-comparison correction"]}
    write_json(Path(output_dir) / "report.json", report)
    return report


def teacher_label_audit(rows, path, output_dir):
    """Teacher/label agreement is descriptive; it never filters the fixed examples."""
    with np.load(path, allow_pickle=False) as arrays:
        qvec, dvec = arrays["queries_vectors"], arrays["documents_vectors"]
        qloc = {key: index for index, key in enumerate(arrays["queries_ids"].tolist())}
        dloc = {key: index for index, key in enumerate(arrays["documents_ids"].tolist())}
    margins = np.asarray([float(qvec[qloc[row["query_id"]]] @
                                (dvec[dloc[row["positive_id"]]] - dvec[dloc[row["negative_id"]]])) for row in rows])
    report = {"rows": len(rows), "teacher_vector_sha256": file_digest(path),
              "positive_minus_bm25_negative_cosine": {"mean": float(margins.mean()),
                                                       "p10": float(np.quantile(margins, .1)),
                                                       "p50": float(np.median(margins)),
                                                       "p90": float(np.quantile(margins, .9))},
              "negative_preferred_count": int((margins < 0).sum()),
              "ties": int((margins == 0).sum()),
              "scope": "Fixed source triplets only. Teacher disagreement is not proof of annotation error; no examples removed or thresholds selected."}
    write_json(Path(output_dir) / "teacher_label_agreement.json", report)
    return report


def student_pilot(rows, teacher_path, base_model_dir, work_dir, output_dir, config, progress):
    from .distill_student import DistillStudent, DistillTrainer
    student = DistillStudent(Path(base_model_dir), student_config(config, 42, "kd_anchor"),
                             prefix_cache_dir=Path(work_dir) / "pilot-prefix")
    queries, documents = source_items(rows)
    items = {"q-" + key: text for key, text in queries.items()}
    items.update({"d-" + key: text for key, text in documents.items()})
    started = time.perf_counter()
    prefix_report = student.cache_prefixes(list(items.values()), list(items), batch_size=8)
    prefix_seconds = time.perf_counter() - started
    equivalence = student.verify_cached_equivalence(list(items.values())[:8], list(items)[:8])
    base_lookup = dict(zip(items, student.base_vectors(list(items.values()), list(items)), strict=True))
    with np.load(teacher_path, allow_pickle=False) as arrays:
        tq, td = arrays["queries_vectors"], arrays["documents_vectors"]
        ql = {key: index for index, key in enumerate(arrays["queries_ids"].tolist())}
        dl = {key: index for index, key in enumerate(arrays["documents_ids"].tolist())}
    trainer = DistillTrainer(student, {"purpose": "discarded timing pilot", "rows": digest(rows)}, [row["query_id"] for row in rows])
    steps = []
    for start in range(0, len(rows), config.batch_size):
        progress.update("student_pilot", completed=start, total=len(rows))
        batch = rows[start:start + config.batch_size]
        begin = time.perf_counter()
        q, d, positive, targets = training_targets(batch, tq, td, ql, dl, student, base_lookup)
        target_seconds = time.perf_counter() - begin
        record = trainer.step(q, d, positive, next_row_cursor=start + len(batch), **targets)
        steps.append({**record, "wall_seconds": time.perf_counter() - begin, "target_preparation_seconds": target_seconds})
    report = {"purpose": "discarded throughput/gradient smoke, not quality evidence", "examples": len(rows),
              "trainable_parameters": student.trainable_parameter_count,
              "prefix": prefix_report, "prefix_wall_seconds": prefix_seconds, "equivalence": equivalence,
              "steps": steps, "median_step_seconds": float(np.median([row["wall_seconds"] for row in steps])),
              "expected_steps_per_full_run": int(np.ceil(config.train_queries / config.batch_size)),
              "expected_runs": len(config.seeds) * len(ARMS), "memory_after": _memory_snapshot(),
              "warning": "Short pilot omits full validation/held-out encoding, thermals, and other machine load"}
    write_json(Path(output_dir) / "pilot_student.json", report)
    return report


def main():
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "register", "run"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/distillation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/distillation"))
    parser.add_argument("--work-dir", type=Path,
                        default=Path(os.environ.get("LOCALAPPDATA", ".")) / "EvidenceForge" / "distillation-v3")
    parser.add_argument("--base-model-dir", type=Path,
                        default=Path("cache/models/1110a243fdf4706b3f48f1d95db1a4f5529b4d41-float"))
    parser.add_argument("--max-hours", type=float, default=11.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = ExperimentConfig()
    progress = Progress(args.output_dir, args.max_hours)
    with single_pipeline(), threadpool_limits(limits=config.threads):
        try:
            prepare(args.data_dir, args.output_dir, config)
            if args.command == "prepare":
                progress.update("prepared")
                return
            train_path, validation_path = source_paths(args.data_dir)
            rows = read_rows(train_path) + read_rows(validation_path)
            if args.command == "pilot":
                selected = sorted(read_rows(train_path), key=lambda row: digest(["timing-pilot", row["query_id"]]))[:64]
                teacher_path, _ = teacher_vectors(selected, args.cache_dir, args.work_dir, args.output_dir, config, progress, pilot=True)
                student_pilot(selected, teacher_path, args.base_model_dir, args.work_dir, args.output_dir, config, progress)
                progress.update("pilot_complete", note="Discarded throughput/gradient smoke; not a training-quality result")
            else:
                frozen_protocol = protocol(args.output_dir, args.data_dir, config, args.base_model_dir)
                if args.command == "register":
                    progress.update("registered", spec_sha256=frozen_protocol["spec_sha256"])
                    return
                teacher_path, _ = teacher_vectors(rows, args.cache_dir, args.work_dir, args.output_dir, config, progress)
                teacher_label_audit(read_rows(train_path), teacher_path, args.output_dir)
                prepare_prefixes(rows, args.base_model_dir, args.work_dir, args.output_dir, config, progress)
                frozen = train_all(args.data_dir, args.base_model_dir, args.work_dir, args.output_dir,
                                   config, progress, frozen_protocol, teacher_path)
                evaluate_students(args.data_dir, args.base_model_dir, args.work_dir, args.output_dir, config, progress, frozen)
                progress.update("complete", report="report.json")
        except BudgetExpired as error:
            write_json(progress.path, {"phase": "budget_exhausted", "last_phase": progress.phase, "error": str(error)})
        except Exception as error:
            write_json(progress.path, {"phase": "failed", "last_phase": progress.phase,
                                       "error": str(error), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == "__main__":
    main()
