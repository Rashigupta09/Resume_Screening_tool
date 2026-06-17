"""End-to-end CLI — the pipeline floor (works even if the UI is cut for time).

    python cli.py --jd data/job_description.txt --cv data/cv_01.txt

Flow: extract -> derive ranking (Call 1) -> scrub CV (local) -> evaluate (Call 2)
-> verify spans + weighted composite (Python) -> write artifacts to out/.
"""
from __future__ import annotations

import argparse
import os
import sys

from src.derive import derive_dimensions, write_ranking_file
from src.evaluate import evaluate_cv, write_evaluation
from src.extract import ExtractionError, extract_text
from src.scrub import candidate_id_from_path, scrub_cv


def run(jd_path: str, cv_path: str, out_dir: str = "out") -> int:
    try:
        jd_text = extract_text(jd_path)
        cv_text = extract_text(cv_path)
    except ExtractionError as e:
        print(f"ERROR (extraction): {e}", file=sys.stderr)
        return 2

    try:
        print("Step 1/4  Deriving ranking dimensions from the JD (Claude) ...")
        rf = derive_dimensions(jd_text, jd_source=os.path.basename(jd_path))
        ranking_path = write_ranking_file(rf, os.path.join(out_dir, "ranking_file.yaml"))
        print(f"          role: {rf.role_title}")
        print(f"          {len(rf.dimensions)} dimensions -> {ranking_path}")

        print("Step 2/4  Scrubbing PII from the CV locally (Presidio, no LLM) ...")
        candidate_id = candidate_id_from_path(cv_path)
        scrub = scrub_cv(cv_text, candidate_id)
        report = ", ".join(f"{k}:{v}" for k, v in scrub.redaction_report.items()) or "none detected"
        print(f"          candidate_id: {candidate_id}  |  redacted: {report}")

        print("Step 3/4  Evaluating the scrubbed CV against the rubric (Claude) ...")
        result = evaluate_cv(rf, scrub.scrubbed_text, candidate_id)

        print("Step 4/4  Verifying evidence spans + weighted composite (Python) ...")
        yaml_path, json_path = write_evaluation(result, out_dir)
    except RuntimeError as e:  # e.g. missing Bedrock API key
        print(f"ERROR (config): {e}", file=sys.stderr)
        return 3
    except Exception as e:  # Bedrock auth/throttle/model errors, JSON parse, etc.
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 4

    _print_summary(rf, result)
    print(f"\nArtifacts: {ranking_path}\n           {yaml_path}\n           {json_path}")
    return 0


def _print_summary(rf, result) -> None:
    name_by_id = {d.id: d.name for d in rf.dimensions}
    print("\n=== Per-dimension scores (primary output) ===")
    for s in result.dimension_scores:
        verified = sum(1 for v in s.verified if v)
        flag = f"{verified}/{len(s.evidence)} evidence verified" if s.evidence else "no evidence"
        print(f"  [{s.score}/5] {name_by_id.get(s.dimension_id, s.dimension_id)}  ({flag})")
        if s.reasoning:
            print(f"        {s.reasoning}")
    print(f"\nWeighted composite (secondary, derived in code): {result.weighted_composite}/5")
    print("\n=== Highlights ===")
    print("  Strengths:        " + "; ".join(result.highlights.strengths))
    print("  Gaps/concerns:    " + "; ".join(result.highlights.gaps_or_concerns))
    print("  Interview probes: " + "; ".join(result.highlights.interview_probes))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Screen one CV against one JD.")
    p.add_argument("--jd", required=True, help="Job description path (.pdf/.txt/.md)")
    p.add_argument("--cv", required=True, help="CV path (.pdf/.txt)")
    p.add_argument("--out", default="out", help="Output directory (default: out)")
    args = p.parse_args(argv)
    return run(args.jd, args.cv, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
