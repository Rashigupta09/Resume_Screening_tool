"""Evidence verification + weighted composite — pure Python, no LLM.

This is the auditable half of the system: it separates LLM *judgment* (the
per-dimension scores) from *arithmetic* (the weighted total), and it guards
against hallucinated evidence by checking each quoted span actually appears in
the scrubbed CV.
"""
from __future__ import annotations

import re
from typing import List

from .schema import Dimension, DimensionScore


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def verify_evidence(spans: List[str], scrubbed_cv: str) -> List[bool]:
    """True where the normalized span is a substring of the normalized CV.

    Spans that don't match are surfaced (not dropped) and flagged as
    unverified / paraphrased in the UI — a hallucination guard.
    """
    haystack = normalize(scrubbed_cv)
    out: List[bool] = []
    for span in spans:
        needle = normalize(span)
        out.append(bool(needle) and needle in haystack)
    return out


def weighted_composite(scores: List[DimensionScore], dimensions: List[Dimension]) -> float:
    """Sum(score * weight) on the 1-5 scale. Computed here, never by the LLM.

    Shown only as secondary, clearly-derived context — never as 'the' verdict.
    """
    weight_by_id = {d.id: d.weight for d in dimensions}
    total = 0.0
    for s in scores:
        total += s.score * weight_by_id.get(s.dimension_id, 0.0)
    return round(total, 2)
