"""Pydantic models — the single source of truth for ranking + evaluation.

Code enforces the *schema* only, never the *content*: dimensions are always
derived from the JD, never hardcoded.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator

WEIGHT_TOLERANCE = 0.02


# --------------------------------------------------------------------------- #
# Ranking file (Call 1 output / Call 2 input)
# --------------------------------------------------------------------------- #
class Dimension(BaseModel):
    id: str
    name: str
    # No upper bound here: the LLM may return percentages (e.g. 30) or raw weights.
    # RankingFile._normalize_weights renormalizes any sum != 1.0 down to fractions.
    weight: float = Field(ge=0.0)
    what_to_look_for: str = ""
    positive_signals: List[str] = Field(default_factory=list)
    negative_signals: List[str] = Field(default_factory=list)


class RankingFile(BaseModel):
    role_title: str
    derived_from: str = ""
    scoring_scale: str = "1-5 (1 = no evidence, 5 = strong direct evidence)"
    dimensions: List[Dimension]
    notes: str = ""

    @model_validator(mode="after")
    def _normalize_weights(self) -> "RankingFile":
        if not self.dimensions:
            raise ValueError("ranking file must contain at least one dimension")
        total = sum(d.weight for d in self.dimensions)
        if total <= 0:
            # JD gave no basis for relative weight -> equal weights, and say so.
            equal = 1.0 / len(self.dimensions)
            for d in self.dimensions:
                d.weight = equal
            self.notes = (self.notes + " Weights equal-weighted (JD gave no basis).").strip()
        elif abs(total - 1.0) > WEIGHT_TOLERANCE:
            # Renormalize so weights sum to 1.0, and record that we did.
            for d in self.dimensions:
                d.weight = d.weight / total
            self.notes = (
                self.notes + f" Weights renormalized to sum to 1.0 (raw sum was {total:.2f})."
            ).strip()
        return self


# --------------------------------------------------------------------------- #
# Evaluation result (Call 2 output, enriched in code)
# --------------------------------------------------------------------------- #
class DimensionScore(BaseModel):
    dimension_id: str
    score: int = Field(ge=1, le=5)
    evidence: List[str] = Field(default_factory=list)
    reasoning: str = ""
    # Filled post-hoc by verify.py (one bool per evidence span). Never from the LLM.
    verified: List[bool] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v):
        """Be robust to an LLM returning 0 / a float / a string for the score."""
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            v = 1
        return max(1, min(5, v))


class Highlights(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    gaps_or_concerns: List[str] = Field(default_factory=list)
    interview_probes: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    candidate_id: str
    evaluated_against: str = "ranking_file.yaml"
    dimension_scores: List[DimensionScore]
    weighted_composite: float = 0.0  # computed in code (verify.py), not by the LLM
    highlights: Highlights = Field(default_factory=Highlights)
