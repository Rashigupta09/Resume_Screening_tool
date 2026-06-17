"""Call 2: evaluate the scrubbed CV against the ranking file.

Per dimension: score (1-5), verbatim evidence, one-line reasoning. Plus
highlights for a non-technical reader. The LLM does NOT compute an overall score
— verify.weighted_composite does, in code. Each evidence span is then verified
against the scrubbed CV (verified / unverified is shown, not hidden).
"""
from __future__ import annotations

import json
import os

import yaml

from .llm import EVAL_MODEL_ID, EVALUATE_SYSTEM, EVALUATE_USER_TEMPLATE, call_claude_json
from .schema import DimensionScore, EvaluationResult, Highlights, RankingFile
from .verify import verify_evidence, weighted_composite


def evaluate_cv(rf: RankingFile, scrubbed_cv: str, candidate_id: str) -> EvaluationResult:
    rubric_json = json.dumps(rf.model_dump(), ensure_ascii=False, indent=2)
    data = call_claude_json(
        EVALUATE_SYSTEM,
        EVALUATE_USER_TEMPLATE.format(ranking_json=rubric_json, scrubbed_cv=scrubbed_cv),
        max_tokens=4096,
        model_id=EVAL_MODEL_ID,
    )

    scores: list[DimensionScore] = []
    for raw in data.get("dimensions", []):
        # The evaluate prompt echoes the dimension as "id"; ranking ids are also "id".
        dim_id = raw.get("dimension_id") or raw.get("id") or ""
        evidence = raw.get("evidence", []) or []
        scores.append(
            DimensionScore(
                dimension_id=dim_id,
                score=raw.get("score", 1),
                evidence=evidence,
                reasoning=raw.get("reasoning", ""),
                verified=verify_evidence(evidence, scrubbed_cv),
            )
        )

    # Guarantee EVERY rubric dimension is scored (acceptance criterion).
    scored_ids = {s.dimension_id for s in scores}
    for d in rf.dimensions:
        if d.id not in scored_ids:
            scores.append(
                DimensionScore(
                    dimension_id=d.id,
                    score=1,
                    evidence=[],
                    reasoning="No assessment returned by the model; defaulted to lowest score.",
                    verified=[],
                )
            )

    h = data.get("highlights", {}) or {}
    highlights = Highlights(
        strengths=h.get("strengths", []) or [],
        gaps_or_concerns=h.get("gaps_or_concerns", []) or [],
        interview_probes=h.get("interview_probes", []) or [],
    )

    return EvaluationResult(
        candidate_id=candidate_id,
        evaluated_against="ranking_file.yaml",
        dimension_scores=scores,
        weighted_composite=weighted_composite(scores, rf.dimensions),
        highlights=highlights,
    )


def write_evaluation(result: EvaluationResult, out_dir: str = "out") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"evaluation_{result.candidate_id}"
    yaml_path = os.path.join(out_dir, stem + ".yaml")
    json_path = os.path.join(out_dir, stem + ".json")
    payload = result.model_dump()
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return yaml_path, json_path
