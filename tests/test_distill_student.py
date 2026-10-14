"""Tiny random six-layer BERT correctness tests; no downloads or real training runs."""
from dataclasses import replace
import hashlib
import json
import random

import numpy as np
import pytest

from evidenceforge.distill_student import DistillConfig, DistillStudent, DistillTrainer, distillation_loss


@pytest.fixture
def tiny_distill_base(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel

    path = tmp_path / "tiny-base"
    path.mkdir()
    torch.manual_seed(319)
    config = BertConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                        num_attention_heads=2, max_position_embeddings=64, pad_token_id=0,
                        hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
    config.to_json_file(path / "config.json")
    model = BertModel(config, add_pooling_layer=False)
    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    state["embeddings.position_ids"] = torch.arange(64).unsqueeze(0)
    save_file(state, str(path / "model.safetensors"))
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "alpha", "beta", "gamma", "delta", "evidence",
                  "study", "treatment", "effect", "unrelated", "long", "document", "query"]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                              special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(path / "tokenizer.json"))
    return path


def config(**overrides):
    values = dict(max_length=24, threads=1, learning_rate=0.002, kd_weight=0, anchor_weight=0)
    values.update(overrides)
    return DistillConfig(**values)


def batch():
    return {"query_texts": ["alpha treatment", "beta effect"],
            "candidate_texts": ["alpha evidence study", "beta evidence", "gamma unrelated document", "delta study"],
            "query_ids": ["q0", "q1"], "candidate_ids": ["d0", "d1", "d2", "d3"],
            "positive_mask": np.array([[True, False, False, False], [False, True, False, False]])}


def test_prefix_cached_forward_full_model_and_base_anchor_equivalence(tiny_distill_base, tmp_path, monkeypatch):
    import torch

    student = DistillStudent(tiny_distill_base, config(), tmp_path / "cache")
    texts, ids = ["alpha", "beta evidence long document", "gamma study"], ["a", "b", "c"]
    summary = student.cache_prefixes(texts, ids, batch_size=3)
    assert summary["new_entries"] == 3 and summary["cache_hits"] == 0 and summary["bytes_written"] > 0
    assert student.trainable_parameter_count == 4448
    assert all(name.startswith(("encoder.layer.4.", "encoder.layer.5.")) for name in student.parameters)
    result = student.verify_cached_equivalence(texts, ids, atol=0)
    assert result["strict_equal"] and result["max_abs_difference"] == 0
    with torch.no_grad():
        inputs = student._inputs(texts)
        full = student.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = torch.nn.functional.normalize((full * mask).sum(1) / mask.sum(1), dim=1)
        actual = student.forward(texts, ids)
        assert torch.allclose(actual, pooled, atol=1e-6, rtol=0)
    base = student.base_vectors(texts, ids)
    assert np.max(np.abs(base - actual.numpy())) < 1e-6

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen prefix/base was recomputed despite a complete cache")

    monkeypatch.setattr(student, "_uncached_prefixes", forbidden)
    assert student.cache_prefixes(texts, ids)["cache_hits"] == 3
    assert np.array_equal(student.base_vectors(texts, ids), base)
    assert np.max(np.abs(student.encode(texts, prefix_ids=ids) - base)) < 1e-6


def test_suffix_updates_freeze_prefix_and_keep_original_anchor(tiny_distill_base, tmp_path):
    import torch

    student = DistillStudent(tiny_distill_base, config(), tmp_path / "cache")
    frozen = {name: value.clone() for name, value in student.model.named_parameters() if not value.requires_grad}
    before = {name: value.clone() for name, value in student.parameters.items()}
    data = batch()
    base = student.base_vectors(data["candidate_texts"], data["candidate_ids"])
    trainer = DistillTrainer(student, {"triples": "fixed-v1"}, ["q0", "q1"])
    record = trainer.step(**data, next_row_cursor=2)
    assert record["gradient_l2_before_clip"] > 0 and not record["dropout_enabled"]
    assert trainer.progress["nonzero_gradient_steps"] == 1
    assert any(not torch.equal(before[name], value) for name, value in student.parameters.items())
    assert all(torch.equal(frozen[name], value) for name, value in student.model.named_parameters() if name in frozen)
    assert all(not layer.training for layer in student.model.modules())
    assert np.array_equal(student.base_vectors(data["candidate_texts"], data["candidate_ids"]), base)
    uncached_base = DistillStudent(tiny_distill_base, config()).base_vectors(data["candidate_texts"])
    assert np.max(np.abs(base - uncached_base)) < 1e-6


def test_candidate_aliases_and_other_known_positives_are_excluded():
    torch = pytest.importorskip("torch")
    query = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], requires_grad=True)
    docs = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]], requires_grad=True)
    positive = np.array([[True, False, True, False], [True, True, False, False]])
    loss = distillation_loss(query, docs, positive, config=config(),
                             candidate_ids=["a", "b", "a", "d"], candidate_texts=["A", "B", "A", "D"])
    assert loss["valid_candidate_mask"].tolist() == [[True, True, False, True], [False, True, False, True]]
    assert loss["target_indices"].tolist() == [0, 1]
    assert loss["unique_candidate_count"] == 3
    # A different ID with identical text aliases the same designated positive.
    aliased = distillation_loss(query[:1], docs, positive[:1], config=config(), primary_positive_indices=[2],
                                candidate_ids=["a", "b", "c", "d"], candidate_texts=["A", "B", "A", "D"])
    assert aliased["target_indices"].tolist() == [0]
    loss["loss"].backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(docs.grad).all()


def test_all_known_positive_pool_is_rejected():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="at least one distinct"):
        distillation_loss(torch.eye(2), torch.eye(2), np.ones((2, 2), dtype=bool), config=config())
    with pytest.raises(ValueError, match="boolean"):
        distillation_loss(torch.eye(2), torch.eye(2), np.eye(2), config=config())


def test_four_objectives_teacher_dimensions_detach_and_unscaled_kl():
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    query = torch.randn(2, 3, requires_grad=True)
    docs = torch.randn(4, 3, requires_grad=True)
    tq, td = torch.randn(2, 7, requires_grad=True), torch.randn(4, 7, requires_grad=True)
    bq, bd = torch.randn(2, 3, requires_grad=True), torch.randn(4, 3, requires_grad=True)
    positive = np.array([[True, False, False, False], [False, True, False, False]])
    values = {}
    for kd_weight, anchor_weight in ((0, 0), (1, 0), (0, 1), (1, 1)):
        values[kd_weight, anchor_weight] = distillation_loss(
            query, docs, positive, config=config(kd_weight=kd_weight, anchor_weight=anchor_weight),
            teacher_query_vectors=tq, teacher_candidate_vectors=td, base_query_vectors=bq, base_candidate_vectors=bd)
    combined = values[1, 1]
    assert torch.allclose(combined["loss"], values[0, 0]["loss"] + values[1, 0]["kd"] + values[0, 1]["anchor"])
    teacher = torch.nn.functional.normalize(tq, dim=1) @ torch.nn.functional.normalize(td, dim=1).T / 0.05
    student = torch.nn.functional.normalize(query, dim=1) @ torch.nn.functional.normalize(docs, dim=1).T / 0.05
    expected = (teacher.softmax(dim=1) * (teacher.log_softmax(dim=1) - student.log_softmax(dim=1))).sum(1).mean()
    assert torch.allclose(combined["kd"], expected, atol=1e-6, rtol=0)
    combined["loss"].backward()
    assert tq.grad is None and td.grad is None and bq.grad is None and bd.grad is None
    assert query.grad.abs().sum() > 0 and docs.grad.abs().sum() > 0


def test_checkpoint_resume_matches_weights_adam_order_and_rng(tiny_distill_base, tmp_path):
    import torch

    settings = config(forward_microbatch_size=2)
    student = DistillStudent(tiny_distill_base, settings, tmp_path / "cache")
    trainer = DistillTrainer(student, {"plan": "fixed"}, ["q0", "q1"])
    trainer.set_epoch_order(["q1", "q0"], epoch=3)
    trainer.step(**batch(), next_row_cursor=1)
    checkpoint = tmp_path / "checkpoint"
    metadata = trainer.save_checkpoint(checkpoint)
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    expected_record = trainer.step(**batch(), next_row_cursor=2)
    expected_weights = {name: parameter.clone() for name, parameter in student.parameters.items()}

    restored = DistillStudent(tiny_distill_base, settings, tmp_path / "cache")
    resumed = DistillTrainer(restored, {"plan": "fixed"}, ["q0", "q1"])
    resumed.load_checkpoint(checkpoint)
    assert resumed.row_order == ["q1", "q0"] and resumed.progress["row_cursor"] == 1
    assert resumed.progress["epoch"] == 3
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected_rng
    actual_record = resumed.step(**batch(), next_row_cursor=2)
    assert actual_record == expected_record
    assert all(torch.equal(expected_weights[name], parameter) for name, parameter in restored.parameters.items())
    assert resumed.progress == trainer.progress
    assert metadata["trainable_parameter_count"] == restored.trainable_parameter_count
    assert metadata["identity"]["student"]["training_mode"] == "eval with gradients; dropout disabled in all blocks"


def test_checkpoint_rejects_wrong_plan_order_and_byte_corruption(tiny_distill_base, tmp_path):
    student = DistillStudent(tiny_distill_base, config())
    trainer = DistillTrainer(student, {"plan": "a"}, ["q0", "q1"])
    trainer.step(**batch())
    directory = tmp_path / "checkpoint"
    metadata = trainer.save_checkpoint(directory)
    other = DistillTrainer(DistillStudent(tiny_distill_base, config()), {"plan": "b"}, ["q0", "q1"])
    with pytest.raises(ValueError, match="identity mismatch"):
        other.load_checkpoint(directory)
    wrong_order = DistillTrainer(DistillStudent(tiny_distill_base, config()), {"plan": "a"}, ["q0", "q2"])
    with pytest.raises(ValueError, match="row order"):
        wrong_order.load_checkpoint(directory)
    state_file = directory / metadata["filename"]
    state_file.write_bytes(state_file.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        trainer.load_checkpoint(directory)


def test_suffix_roundtrip_and_base_identity(tiny_distill_base, tmp_path):
    import torch

    student = DistillStudent(tiny_distill_base, config())
    DistillTrainer(student, {"plan": 1}, ["q0", "q1"]).step(**batch())
    directory = tmp_path / "suffix"
    metadata = student.save_adapter(directory)
    assert metadata["parameter_sha256"] != metadata["initial_parameter_sha256"]
    restored = DistillStudent(tiny_distill_base, config())
    restored.load_adapter(directory)
    assert all(torch.equal(value, restored.parameters[name]) for name, value in student.parameters.items())
    incompatible = DistillStudent(tiny_distill_base, replace(config(), max_length=20))
    with pytest.raises(ValueError, match="identity"):
        incompatible.load_adapter(directory)


def test_prefix_identity_checksum_and_cross_seed_reuse(tiny_distill_base, tmp_path):
    student = DistillStudent(tiny_distill_base, config(), tmp_path / "cache")
    student.cache_prefixes(["alpha"], ["a"])
    second = DistillStudent(tiny_distill_base, replace(config(), seed=99, kd_weight=1, forward_microbatch_size=1), tmp_path / "cache")
    assert second.cache_root == student.cache_root
    assert second.identity != student.identity
    assert second.cache_prefixes(["alpha"], ["a"])["cache_hits"] == 1
    assert second.cache_prefixes(["beta"], ["a"])["new_entries"] == 1
    path, _ = student._entry("alpha", "a")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["hidden"][0, 0] += 0.5
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        student.get_prefixes(["alpha"], ["a"])


def test_zero_step_checkpoint_and_json_identity_validation(tiny_distill_base, tmp_path):
    student = DistillStudent(tiny_distill_base, config())
    trainer = DistillTrainer(student, {"plan": 1}, ["q0", "q1"])
    trainer.save_checkpoint(tmp_path / "zero")
    trainer.load_checkpoint(tmp_path / "zero")
    assert trainer.progress["completed_steps"] == 0
    path = tmp_path / "zero" / "checkpoint.json"
    record = json.loads(path.read_text())
    record["row_order"] = ["q1", "q0"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="row order"):
        trainer.load_checkpoint(tmp_path / "zero")


def test_cached_batch_reads_unique_files_once_and_keeps_outputs_independent(tiny_distill_base, tmp_path, monkeypatch):
    student = DistillStudent(tiny_distill_base, config(), tmp_path / "cache")
    student.cache_prefixes(["alpha", "beta evidence"], ["a", "b"])
    original, calls = student._read_entry, []

    def counted(path, identity, **kwargs):
        calls.append(path)
        return original(path, identity, **kwargs)

    monkeypatch.setattr(student, "_read_entry", counted)
    values = student.get_prefixes(["alpha", "beta evidence", "alpha"], ["a", "b", "a"])
    assert len(calls) == len(set(calls)) == 2
    assert np.array_equal(values[0], values[2]) and not np.shares_memory(values[0], values[2])
    retained = values[2].copy()
    values[0][0, 0] += 123
    assert np.array_equal(values[2], retained)
    assert np.array_equal(student.get_prefixes(["alpha"], ["a"])[0], retained)
    calls.clear()
    base = student.base_vectors(["alpha", "beta evidence", "alpha"], ["a", "b", "a"])
    assert len(calls) == len(set(calls)) == 2 and np.array_equal(base[0], base[2])
    calls.clear()
    assert student.cache_prefixes(["alpha", "alpha", "beta evidence"], ["a", "a", "b"])["cache_hits"] == 3
    assert len(calls) == len(set(calls)) == 2


def test_partial_cached_batch_builds_missing_entries_without_rereading_existing(tiny_distill_base, tmp_path, monkeypatch):
    student = DistillStudent(tiny_distill_base, config(), tmp_path / "cache")
    student.cache_prefixes(["alpha"], ["a"])
    existing, _ = student._entry("alpha", "a")
    original, calls = student._read_entry, []

    def counted(path, identity, **kwargs):
        calls.append(path)
        return original(path, identity, **kwargs)

    monkeypatch.setattr(student, "_read_entry", counted)
    values = student.get_prefixes(["alpha", "beta evidence", "alpha"], ["a", "b", "a"])
    assert calls.count(existing) == 1
    assert len(values) == 3 and all(value.dtype == np.float32 for value in values)
    assert student.verify_cached_equivalence(["alpha", "beta evidence"], ["a", "b"])["ok"]


def test_buffer_hash_preserves_existing_c_order_cache_checksums():
    from evidenceforge.distill_student import _array_sha

    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    for value in (data, np.asfortranarray(data), data[::2], np.array(1, dtype=np.float32)):
        assert _array_sha(value) == hashlib.sha256(value.tobytes()).hexdigest()


def test_length_microbatches_match_full_padding_vectors_loss_and_gradients(tiny_distill_base):
    import torch

    settings = config(kd_weight=1, anchor_weight=1, forward_microbatch_size=2)
    student = DistillStudent(tiny_distill_base, settings)
    queries = ["alpha treatment", "beta evidence long document study effect query treatment", "gamma"]
    documents = ["alpha evidence " + "long " * 18, "beta", "gamma study " + "document " * 8,
                 "delta unrelated study", "beta effect " + "treatment " * 14, "gamma unrelated document evidence study"]
    prefixes = student.get_prefixes(queries + documents, use_cache=False)
    qp, dp = prefixes[:3], prefixes[3:]
    with torch.no_grad():
        base_q, base_d = student._forward_prefix_group(qp), student._forward_prefix_group(dp)
    torch.manual_seed(778)
    targets = {"positive_mask": np.array([[True, False, False, True, False, False],
                                         [False, True, False, False, False, False],
                                         [False, False, True, False, False, False]]),
               "config": settings, "teacher_query_vectors": torch.randn(3, 7),
               "teacher_candidate_vectors": torch.randn(6, 7),
               "base_query_vectors": base_q, "base_candidate_vectors": base_d,
               "candidate_ids": [f"d{index}" for index in range(6)]}
    full_q, full_d = student._forward_prefix_group(qp), student._forward_prefix_group(dp)
    full = distillation_loss(full_q, full_d, **targets)
    full["loss"].backward()
    gradients = {name: parameter.grad.clone() for name, parameter in student.parameters.items()}
    student.model.zero_grad(set_to_none=True)
    grouped_q, grouped_d = student.forward_prefixes(qp), student.forward_prefixes(dp)
    grouped = distillation_loss(grouped_q, grouped_d, **targets)
    assert torch.allclose(grouped_q, full_q, atol=1e-6, rtol=0)
    assert torch.allclose(grouped_d, full_d, atol=1e-6, rtol=0)
    assert torch.allclose(grouped["loss"], full["loss"], atol=1e-5, rtol=0)
    assert torch.equal(grouped["valid_candidate_mask"], full["valid_candidate_mask"])
    assert torch.equal(grouped["target_indices"], full["target_indices"])
    assert grouped["target_indices"].tolist() == [0, 1, 2]
    assert grouped["unique_candidate_count"] == 6
    grouped["loss"].backward()
    assert all(torch.allclose(parameter.grad, gradients[name], atol=2e-6, rtol=1e-4)
               for name, parameter in student.parameters.items())
    assert student.last_forward_stats["microbatches"] == 3
    assert student.last_forward_stats["padded_tokens"] < student.last_forward_stats["full_batch_padded_tokens"]


def test_entropy_diagnostics_use_full_masked_pool_even_without_kd():
    torch = pytest.importorskip("torch")
    query = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], requires_grad=True)
    docs = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]], requires_grad=True)
    teacher_q = torch.tensor([[1.0, 0], [0, 1.0]])
    teacher_d = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [0.6, 0.8]])
    kwargs = {"positive_mask": np.array([[True, False, True, False], [True, True, False, False]]),
              "config": config(), "candidate_ids": ["a", "b", "a", "d"]}
    absent = distillation_loss(query, docs, **kwargs)
    supplied = distillation_loss(query, docs, **kwargs, teacher_query_vectors=teacher_q,
                                 teacher_candidate_vectors=teacher_d)
    assert torch.equal(absent["loss"], supplied["loss"]) and float(supplied["kd"]) == 0
    for name in ("teacher_distribution_entropy", "student_distribution_entropy", "teacher_primary_positive_probability"):
        assert absent[name] is None and math_is_finite(supplied[name])
    scores = teacher_q @ teacher_d.T / 0.05
    scores = scores.masked_fill(~supplied["valid_candidate_mask"], torch.finfo(scores.dtype).min)
    probabilities, logs = scores.softmax(1), scores.log_softmax(1)
    expected_entropy = float(-(probabilities * logs).sum(1).mean())
    expected_primary = float(probabilities[torch.arange(2), supplied["target_indices"]].mean())
    assert supplied["teacher_distribution_entropy"] == pytest.approx(expected_entropy, abs=1e-7)
    assert supplied["teacher_primary_positive_probability"] == pytest.approx(expected_primary, abs=1e-7)


def math_is_finite(value):
    return isinstance(value, float) and bool(np.isfinite(value))


@pytest.mark.parametrize("settings", [{"kd_temperature": 0}, {"anchor_weight": -1}, {"seed": -1}, {"threads": 0},
                                     {"forward_microbatch_size": 0}])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        config(**settings).validate()
