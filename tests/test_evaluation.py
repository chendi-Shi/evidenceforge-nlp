import math

import pytest

from evidenceforge.evaluation import candidate_ceiling, macro_metrics, paired_bootstrap, query_metrics


def test_metrics_reward_rank_and_do_not_double_count_duplicates():
    result = query_metrics(["noise", "a", "a", "b"], {"a": 1, "b": 1})
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert result["ndcg@10"] == pytest.approx(expected)
    assert result["mrr@10"] == 0.5
    assert result["recall@10"] == 1.0


def test_cutoffs_and_missing_gold_are_not_normalized_away():
    ranking = [f"noise{i}" for i in range(10)] + ["gold"]
    result = query_metrics(ranking, {"gold": 1, "missing": 1})
    assert result == {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@10": 0.0, "recall@50": 0.5}
    oracle = candidate_ceiling(ranking, {"gold": 1, "missing": 1})
    assert oracle["oracle_ndcg@10"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert oracle["candidate_recall@50"] == 0.5


def test_graded_relevance_uses_exponential_gain():
    result = query_metrics(["b", "a"], {"a": 2, "b": 1})
    assert result["ndcg@10"] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))


def test_paired_bootstrap_zero_difference_and_reproducibility():
    values = [0.0, 0.5, 1.0]
    same = paired_bootstrap(values, values, samples=200)
    assert same["delta"] == same["ci95_low"] == same["ci95_high"] == 0.0
    a = paired_bootstrap([0.2, 0.7, 1.0], values, samples=200)
    assert a == paired_bootstrap([0.2, 0.7, 1.0], values, samples=200)
    assert a["ci95_low"] >= 0


def test_invalid_empty_or_mismatched_evaluation_fails():
    with pytest.raises(ValueError):
        macro_metrics([])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [])
