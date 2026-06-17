"""Schema validation: weight normalization + score clamping (pure, no LLM/AWS)."""
import math

from src.schema import DimensionScore, RankingFile


def _ranking(weights):
    return {
        "role_title": "Test Role",
        "dimensions": [
            {"id": f"dim_{i}", "name": f"D{i}", "weight": w, "what_to_look_for": "x"}
            for i, w in enumerate(weights, start=1)
        ],
    }


def test_weights_already_sum_to_one_are_preserved():
    rf = RankingFile.model_validate(_ranking([0.5, 0.3, 0.2]))
    assert math.isclose(sum(d.weight for d in rf.dimensions), 1.0, abs_tol=1e-6)
    assert rf.dimensions[0].weight == 0.5


def test_weights_are_renormalized_when_they_do_not_sum_to_one():
    rf = RankingFile.model_validate(_ranking([3, 1, 1]))  # sum = 5
    assert math.isclose(sum(d.weight for d in rf.dimensions), 1.0, abs_tol=1e-6)
    assert math.isclose(rf.dimensions[0].weight, 0.6, abs_tol=1e-6)
    assert "renormalized" in rf.notes


def test_zero_weights_fall_back_to_equal_and_say_so():
    rf = RankingFile.model_validate(_ranking([0, 0, 0, 0]))
    assert all(math.isclose(d.weight, 0.25, abs_tol=1e-6) for d in rf.dimensions)
    assert "equal-weighted" in rf.notes


def test_score_is_clamped_into_1_to_5():
    assert DimensionScore(dimension_id="d", score=0).score == 1
    assert DimensionScore(dimension_id="d", score=7).score == 5
    assert DimensionScore(dimension_id="d", score="3").score == 3
    assert DimensionScore(dimension_id="d", score=4.6).score == 5
