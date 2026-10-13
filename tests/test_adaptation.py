"""Tiny real transformer gradient tests, without network or downloaded weights."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evidenceforge.adaptation import AdaptationConfig, build_adaptation_plan


class TrainOnlyQrels(dict):
    def __getitem__(self, key):
        if key != "train":
            raise AssertionError("Neural training attempted to access test judgments")
        return super().__getitem__(key)


class MiningIndex:
    metadata = {"model_id": "fixture-frozen-teacher", "model_revision": "fixture-v1"}

    def __init__(self, dataset):
        self.dataset = dataset

    def search(self, query, method="rrf", k=50):
        own = query.split()[-1]
        scores = {doc["_id"]: (0.9 if doc["_id"] == own else 0.1) for doc in self.dataset.docs}
        # One high-scoring unlabeled candidate exercises potential-false-negative filtering.
        decoy = next(doc["_id"] for doc in self.dataset.docs if doc["_id"] != own)
        scores[decoy] = 0.88
        ordered = sorted(self.dataset.docs, key=lambda doc: (-scores[doc["_id"]], doc["_id"]))
        return [{"doc_id": doc["_id"], "dense_score": scores[doc["_id"]], "title": doc["title"], "text": doc["text"]}
                for doc in ordered[:k]]


def dataset_fixture():
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    docs = [{"_id": name, "title": name, "text": f"Scientific evidence about {name} treatment response"} for name in names]
    queries = {name: f"Evidence supporting {name}" for name in names}
    qrels = TrainOnlyQrels({"train": {name: {name: 1} for name in names}})
    return SimpleNamespace(docs=docs, queries=queries, qrels=qrels)


def test_negative_arms_match_budget_and_never_read_test():
    data = dataset_fixture()
    config = AdaptationConfig(max_train_queries=4, negatives_per_query=2)
    plan = build_adaptation_plan(data, MiningIndex(data), config)
    assert not set(plan["train_query_ids"]) & set(plan["validation_query_ids"])
    assert plan["split"]["shared_positive_documents"] == 0
    assert plan["mining_statistics"]["teacher_excluded"] > 0
    for rows in zip(*(plan["arms"][name] for name in ("random", "retrieved", "filtered"))):
        assert len({(row["query_id"], row["positive_doc_id"]) for row in rows}) == 1
        for row in rows:
            assert len(set(row["negative_doc_ids"])) == config.negatives_per_query
            assert row["positive_doc_id"] not in row["negative_doc_ids"]
        assert rows[1]["negative_doc_ids"][0] not in rows[2]["negative_doc_ids"]
    repeated = build_adaptation_plan(data, MiningIndex(data), config)
    assert plan["plan_sha256"] == repeated["plan_sha256"]
    with pytest.raises(ValueError, match="Teacher score missing"):
        build_adaptation_plan(data, MiningIndex(data), config, teacher_scores={})


@pytest.fixture
def tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel

    path = tmp_path / "base"
    path.mkdir()
    torch.manual_seed(123)
    config = BertConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, max_position_embeddings=64, pad_token_id=0)
    model = BertModel(config, add_pooling_layer=False)
    config.to_json_file(path / "config.json")
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    # Match the legacy persistent position_ids present in the real MiniLM checkpoint.
    state["embeddings.position_ids"] = torch.arange(config.max_position_embeddings).unsqueeze(0)
    save_file(state, str(path / "model.safetensors"))
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "scientific", "evidence", "about", "treatment", "response", "supporting",
              "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(path / "tokenizer.json"))
    return path


def tiny_config(**kwargs):
    values = dict(max_train_queries=4, epochs=1, batch_size=2, negatives_per_query=1, max_length=32,
                  lora_rank=2, lora_alpha=4.0, learning_rate=0.005, threads=1, checkpoint_every=1)
    values.update(kwargs)
    return AdaptationConfig(**values)


def test_real_gradients_artifact_roundtrip_and_shared_prefix_equivalence(tiny_model, tmp_path):
    from evidenceforge.adaptation import AdaptedEncoder, MultiAdapterEncoder, train_adaptation
    data, config = dataset_fixture(), tiny_config()
    output = tmp_path / "run"
    report = train_adaptation(data, MiningIndex(data), tiny_model, output, config)
    assert set(report["arms"]) == {"random", "retrieved", "filtered"}
    assert len({arm["initial_trainable_sha256"] for arm in report["arms"].values()}) == 1
    for arm in report["arms"].values():
        assert arm["completed_steps"] == 2
        assert arm["nonzero_gradient_steps"] > 0
        assert arm["parameter_delta_l2"] > 0
        assert arm["initial_trainable_sha256"] != arm["final_trainable_sha256"]
        assert arm["trainable_parameter_count"] == 128
        assert all("lora_" in name for name in arm["trainable_parameter_names"])
    texts = ["Evidence supporting alpha", "Scientific evidence about bravo treatment response", "charlie"]
    combined = MultiAdapterEncoder(tiny_model, {arm: output / arm for arm in report["arms"]}, config)
    variants = combined.encode(texts, batch_size=2)
    # Mixed padding lengths and a second invocation also catch stale adapter-state leakage.
    repeated = combined.encode(texts, batch_size=2)
    base = AdaptedEncoder(tiny_model, config).encode(texts, batch_size=2)
    np.testing.assert_allclose(variants["base"], base, rtol=1e-5, atol=1e-6)
    for name in report["arms"]:
        independent = AdaptedEncoder.load(tiny_model, output / name, threads=1).encode(texts, batch_size=2)
        np.testing.assert_allclose(variants[name], independent, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(repeated[name], independent, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(variants[name], axis=1), 1, atol=1e-6)
    reused = train_adaptation(data, MiningIndex(data), tiny_model, output, config)
    assert reused["arms"]["random"]["adapter_sha256"] == report["arms"]["random"]["adapter_sha256"]
    with pytest.raises(ValueError, match="different training plan"):
        train_adaptation(data, MiningIndex(data), tiny_model, output, replace(config, epochs=2))


def test_interrupted_resume_matches_uninterrupted_weights(tiny_model, tmp_path):
    from evidenceforge.adaptation import train_adaptation
    data, config = dataset_fixture(), tiny_config(arms=("retrieved",))

    def interrupt(arm, step):
        if step == 1:
            raise InterruptedError("deliberate interruption after persisted checkpoint")

    with pytest.raises(InterruptedError):
        train_adaptation(data, MiningIndex(data), tiny_model, tmp_path / "resumed", config, progress_callback=interrupt)
    resumed = train_adaptation(data, MiningIndex(data), tiny_model, tmp_path / "resumed", config)
    uninterrupted = train_adaptation(data, MiningIndex(data), tiny_model, tmp_path / "fresh", config)
    assert resumed["arms"]["retrieved"]["final_trainable_sha256"] == uninterrupted["arms"]["retrieved"]["final_trainable_sha256"]
    assert resumed["arms"]["retrieved"]["trace"] == uninterrupted["arms"]["retrieved"]["trace"]
    checkpoint = json.loads((tmp_path / "resumed" / "retrieved" / "latest.json").read_text())["checkpoint"]
    assert (tmp_path / "resumed" / "retrieved" / "checkpoints" / f"{checkpoint}.safetensors").exists()
    assert not list(Path(tmp_path).rglob("*.pt"))
