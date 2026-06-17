"""Evidence verification + weighted composite (pure Python, the auditable half)."""
import math

from src.schema import Dimension, DimensionScore
from src.verify import normalize, verify_evidence, weighted_composite

CV = "Led the payments service serving ~12,000 requests per day.\nMentored two engineers."


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello   WORLD\n\t ") == "hello world"


def test_verbatim_span_is_verified():
    assert verify_evidence(["Led the payments service"], CV) == [True]


def test_whitespace_only_difference_still_verifies():
    # normalized substring match tolerates differing whitespace/newlines
    assert verify_evidence(["payments   service\nserving"], CV) == [True]


def test_paraphrased_or_absent_span_is_flagged_unverified():
    assert verify_evidence(["Ran the billing platform"], CV) == [False]
    assert verify_evidence([""], CV) == [False]


def test_weighted_composite_arithmetic():
    dims = [Dimension(id="a", name="A", weight=0.6), Dimension(id="b", name="B", weight=0.4)]
    scores = [
        DimensionScore(dimension_id="a", score=5),
        DimensionScore(dimension_id="b", score=3),
    ]
    # 5*0.6 + 3*0.4 = 4.2
    assert math.isclose(weighted_composite(scores, dims), 4.2, abs_tol=1e-6)


def test_weighted_composite_ignores_unknown_dimension_ids():
    dims = [Dimension(id="a", name="A", weight=1.0)]
    scores = [
        DimensionScore(dimension_id="a", score=4),
        DimensionScore(dimension_id="ghost", score=5),
    ]
    assert math.isclose(weighted_composite(scores, dims), 4.0, abs_tol=1e-6)
