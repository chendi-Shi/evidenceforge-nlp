"""Tiny real-gradient runner integration; no public datasets or models are loaded."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from evidenceforge import distill_eval, distill_experiment as experiment, distill_student
from evidenceforge.data import Dataset
from evidenceforge.distill_data import document_id, query_id


@pytest.fixture
def tiny_runner_base(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel

    directory = tmp_path / "tiny-base"
    directory.mkdir()
    torch.manual_seed(319)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "alpha", "beta", "gamma", "delta", "evidence",
                  "study", "treatment", "effect", "unrelated", "long", "document", "query", "epsilon", "zeta"]
    config = BertConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                        num_hidden_layers=6, num_attention_heads=2, max_position_embeddings=64,
                        pad_token_id=0, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
    config.to_json_file(directory / "config.json")
    model = BertModel(config, add_pooling_layer=False)
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    state["embeddings.position_ids"] = torch.arange(64).unsqueeze(0)
    save_file(state, str(directory / "model.safetensors"))
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(vocabulary)},
                                           unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                              special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(directory / "tokenizer.json"))
    return directory


@dataclass
class TinyRun:
    data: Path
    base: Path
    work: Path
    output: Path
    teacher: Path
    config: experiment.ExperimentConfig
    protocol: dict
    rows: list[dict]

    def train(self, progress=None):
        return experiment.train_all(self.data, self.base, self.work, self.output, self.config,
                                    progress or experiment.Progress(self.output, 1), self.protocol, self.teacher)


def _make_run(directory, base, monkeypatch):
    class MetadataOnlyTeacher:
        def __init__(self, *args, **kwargs):
            self.metadata = {"model": "synthetic-seven-dimensional-teacher", "tokens": kwargs["max_tokens"]}

    monkeypatch.setattr(experiment, "QwenEmbeddingEncoder", MetadataOnlyTeacher)
    data, output, work = directory / "data", directory / "results", directory / "work"
    prepared = data / "prepared"
    prepared.mkdir(parents=True)
    rows = []
    for index, word in enumerate(("alpha", "beta", "gamma", "delta", "epsilon", "zeta")):
        query, positive, negative = f"{word} treatment", f"{word} evidence study", f"unrelated {word} document"
        rows.append({"query_id": query_id(query), "query": query,
                     "positive_id": document_id(positive), "positive": positive,
                     "negative_id": document_id(negative), "negative": negative,
                     "known_positive_ids": [document_id(positive)], "source_row_index": index})
    # A second judged positive in the validation pool must survive qrels creation.
    rows[4]["known_positive_ids"].append(rows[5]["positive_id"])
    for name, values in (("train", rows[:4]), ("validation", rows[4:])):
        (prepared / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
    config = experiment.ExperimentConfig(train_queries=4, validation_queries=2, seeds=(17,),
                                         student_tokens=32, teacher_tokens=32, batch_size=2,
                                         learning_rate=0.002, validation_every=1, threads=1)
    queries, documents = experiment.source_items(rows)
    rng = np.random.default_rng(313)

    def unit_vectors(count):
        values = rng.normal(size=(count, 7)).astype(np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    work.mkdir(parents=True)
    teacher = work / "synthetic-teacher.npz"
    np.savez(teacher, queries_ids=np.asarray(list(queries)), documents_ids=np.asarray(list(documents)),
             queries_vectors=unit_vectors(len(queries)), documents_vectors=unit_vectors(len(documents)))
    frozen_protocol = experiment.protocol(output, data, config, base)
    return TinyRun(data, base, work, output, teacher, config, frozen_protocol, rows)


@pytest.fixture
def tiny_run(tmp_path, tiny_runner_base, monkeypatch):
    # The actual loader stays forbidden until an individual test explicitly
    # supplies the tiny held-out Dataset after all trained weights are frozen.
    def no_held_out_access(*args, **kwargs):
        pytest.fail("Held-out data was accessed before the test unsealed it")

    monkeypatch.setattr(distill_eval, "load_or_download_scidocs", no_held_out_access)
    return _make_run(tmp_path / "run", tiny_runner_base, monkeypatch)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tiny_scidocs():
    return Dataset(
        docs=[{"_id": "held-alpha", "title": "alpha study", "text": "alpha evidence"},
              {"_id": "held-beta", "title": "beta study", "text": "beta evidence"},
              {"_id": "held-negative", "title": "unrelated", "text": "gamma document"}],
        queries={"target-a": "alpha query", "target-b": "beta query"},
        qrels={"test": {"target-a": {"held-alpha": 1}, "target-b": {"held-beta": 1}}},
        metadata={"dataset": "synthetic-test-fixture", "documents": 3, "queries": 2},
    )


def test_four_arms_real_gradients_shared_validation_pool_and_post_freeze_evaluation(tiny_run, monkeypatch):
    run = tiny_run
    prefix = experiment.prepare_prefixes(run.rows, run.base, run.work, run.output, run.config,
                                         experiment.Progress(run.output, 1))
    assert prefix["equivalence"]["ok"]
    frozen = run.train()
    assert set(frozen["runs"]) == {f"{arm}/17" for arm in ("ce", "kd", "anchor", "kd_anchor")}
    assert _json(run.output / "frozen_students.json") == frozen
    assert _json(run.output / "protocol.json")["spec_sha256"] == frozen["spec_sha256"]
    orders = set()
    for key, report in frozen["runs"].items():
        arm, seed = key.split("/")
        assert report["progress"] == {"completed_steps": 2, "nonzero_gradient_steps": 2, "epoch": 0, "row_cursor": 4}
        trace = _json(run.output / "training" / key / "trace.json")
        assert [row["step"] for row in trace] == [1, 2]
        assert all(row["gradient_l2_before_clip"] > 0 and not row["dropout_enabled"] for row in trace)
        assert all(np.isfinite(row["loss"]) for row in trace)
        assert report["final_suffix"]["parameter_sha256"] != report["final_suffix"]["initial_parameter_sha256"]
        assert report["selection"]["best_step"] in {1, 2}
        assert [row["step"] for row in report["selection"]["validations"]] == [1, 2]
        checkpoint = _json(run.work / "checkpoints" / arm / seed / "checkpoint.json")
        assert checkpoint["progress"]["row_cursor"] == 4
        assert checkpoint["progress"]["completed_steps"] == 2
        assert (run.work / "checkpoints" / arm / seed / checkpoint["filename"]).is_file()
        orders.add(report["plan"]["order_sha256"])
    assert len(orders) == 1  # Arm comparisons see exactly the same minibatch order.

    validation = _json(run.output / "source_validation_base.json")
    assert validation["query_count"] == 2 and validation["corpus_size"] == 4
    assert "not the 8.8M" in validation["scope"]
    expected_queries = {row["query_id"] for row in run.rows[4:]}
    expected_docs = {row[key] for row in run.rows[4:] for key in ("positive_id", "negative_id")}
    paths = [run.output / "validation/base.trec", *run.output.glob("training/*/*/validation/step-*.trec")]
    assert len(paths) == 9
    for path in paths:
        pools = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            qid, _, docid, *_ = line.split()
            pools.setdefault(qid, set()).add(docid)
        assert set(pools) == expected_queries
        assert all(pool == expected_docs for pool in pools.values())
    assert len(experiment.validation_items(run.rows[4:])[2][run.rows[4]["query_id"]]) == 2

    calls = []

    def tiny_loader(path):
        assert (run.output / "frozen_students.json").is_file()
        calls.append(path)
        return _tiny_scidocs()

    monkeypatch.setattr(distill_eval, "load_or_download_scidocs", tiny_loader)
    result = experiment.evaluate_students(run.data, run.base, run.work, run.output, run.config,
                                          experiment.Progress(run.output, 1), frozen)
    assert len(calls) == 1
    assert set(result["runs"]) == {"base", "ce", "kd", "anchor", "kd_anchor"}
    assert result["summary"]["primary"]["treatment"] == "kd_anchor"
    for seed_runs in result["runs"].values():
        for report in seed_runs.values():
            assert report["query_count"] == 2 and report["corpus_size"] == 3
            assert {row["query_id"] for row in report["per_query"]} == {"target-a", "target-b"}

    # Completed training must verify and reuse weights, without doing more
    # optimizer steps or even constructing another student model.
    def forbidden_student(*args, **kwargs):
        pytest.fail("A completed arm was trained or encoded again during resume")

    monkeypatch.setattr(distill_student, "DistillStudent", forbidden_student)
    assert run.train() == frozen


def test_selected_weight_tampering_is_rejected_before_resume_or_heldout_access(tiny_run):
    frozen = tiny_run.train()
    selected = tiny_run.output / "training/ce/17/selected"
    metadata = _json(selected / "metadata.json")
    weights = selected / metadata["filename"]
    weights.write_bytes(weights.read_bytes() + b"altered-after-freeze")
    with pytest.raises(ValueError, match="weights changed"):
        tiny_run.train()
    with pytest.raises(ValueError, match="weights changed"):
        experiment.evaluate_students(tiny_run.data, tiny_run.base, tiny_run.work, tiny_run.output,
                                     tiny_run.config, experiment.Progress(tiny_run.output, 1), frozen)


def test_incomplete_freeze_cannot_access_heldout_dataset(tiny_run):
    partial = {"spec_sha256": tiny_run.protocol["spec_sha256"], "runs": {"ce/17": {}}}
    with pytest.raises(ValueError, match="before every configured student is frozen"):
        experiment.evaluate_students(tiny_run.data, tiny_run.base, tiny_run.work, tiny_run.output,
                                     tiny_run.config, experiment.Progress(tiny_run.output, 1), partial)


def test_budget_expiry_preserves_atomic_batch_and_resumes_identically(tiny_run, tmp_path, monkeypatch):
    class ExpireAfterFirstBatch(experiment.Progress):
        def update(self, phase, **fields):
            if phase == "training" and fields["examples_completed"] == 2:
                self.deadline = -1
            super().update(phase, **fields)

    with pytest.raises(experiment.BudgetExpired):
        tiny_run.train(ExpireAfterFirstBatch(tiny_run.output, 1))
    partial_checkpoint = _json(tiny_run.work / "checkpoints/ce/17/checkpoint.json")
    assert partial_checkpoint["progress"]["completed_steps"] == 1
    assert partial_checkpoint["progress"]["row_cursor"] == 2
    assert [row["step"] for row in _json(tiny_run.output / "training/ce/17/trace.json")] == [1]
    assert not (tiny_run.output / "frozen_students.json").exists()
    resumed = tiny_run.train()
    uninterrupted = _make_run(tmp_path / "uninterrupted", tiny_run.base, monkeypatch)
    direct = uninterrupted.train()
    for key, report in resumed["runs"].items():
        comparison = direct["runs"][key]
        assert report["progress"] == comparison["progress"]
        assert report["selection"] == comparison["selection"]
        assert report["final_suffix"]["parameter_sha256"] == comparison["final_suffix"]["parameter_sha256"]
        assert _json(tiny_run.output / "training" / key / "trace.json") == _json(
            uninterrupted.output / "training" / key / "trace.json")
        # This checksum covers suffix tensors, Adam moments and the torch RNG.
        a = _json(tiny_run.work / "checkpoints" / key / "checkpoint.json")
        b = _json(uninterrupted.work / "checkpoints" / key / "checkpoint.json")
        assert a["filename"] == b["filename"]
        assert a["rng_python"] == b["rng_python"] and a["rng_numpy"] == b["rng_numpy"]
