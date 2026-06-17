"""Streamlit UI — a thin render layer over src/. The four panels mirror the
pipeline so the coordinator can follow the reasoning end to end:

  1. Ranking file  (how we'll judge — shown BEFORE evaluation)
  2. Redaction report + scrubbed preview (PII removed locally)
  3. Per-dimension scorecard (evidence + verified/unverified, composite secondary)
  4. Highlights (strengths / gaps / interview probes, plain language)
"""
from __future__ import annotations

import json
import os
import tempfile

import yaml
import streamlit as st
from dotenv import load_dotenv

from src.derive import derive_dimensions, write_ranking_file
from src.evaluate import evaluate_cv, write_evaluation
from src.extract import ExtractionError, extract_text
from src.scrub import candidate_id_from_path, scrub_cv

load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR", "data")

st.set_page_config(page_title="Resume Screening Tool", layout="wide")
st.title("Resume Screening Tool")
st.caption(
    "Screen one CV against one JD. Every score traces back to a JD-derived dimension; "
    "every quote is verified against the CV. PII is scrubbed locally before any LLM call."
)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def _materialize(upload) -> str:
    """Persist a Streamlit upload to a temp file so extract_text can read it by path."""
    suffix = os.path.splitext(upload.name)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.close()
    return tmp.name


def _pick(label: str, key: str) -> tuple[str | None, str | None]:
    """Return (path, display_name) from either an upload or a file in INPUT_DIR."""
    upload = st.file_uploader(f"Upload {label}", type=["pdf", "txt", "md"], key=f"up_{key}")
    if upload is not None:
        return _materialize(upload), upload.name

    choices = []
    if os.path.isdir(INPUT_DIR):
        choices = sorted(
            f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith((".pdf", ".txt", ".md"))
        )
    if choices:
        chosen = st.selectbox(f"…or pick {label} from {INPUT_DIR}/", ["—"] + choices, key=f"sel_{key}")
        if chosen and chosen != "—":
            return os.path.join(INPUT_DIR, chosen), chosen
    return None, None


def _cell(text: str) -> str:
    """Make a string safe for a one-line markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


with st.sidebar:
    st.header("Inputs")
    jd_path, jd_name = _pick("Job Description", "jd")
    st.divider()
    cv_path, cv_name = _pick("CV", "cv")
    st.divider()
    run = st.button("Run screening", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Pipeline (cached in session_state so downloads don't re-run Claude)
# --------------------------------------------------------------------------- #
if run:
    if not jd_path or not cv_path:
        st.error("Select/upload both a Job Description and a CV first.")
        st.stop()
    try:
        with st.spinner("Extracting text …"):
            jd_text = extract_text(jd_path)
            cv_text = extract_text(cv_path)
        with st.spinner("Deriving ranking dimensions from the JD (Claude) …"):
            rf = derive_dimensions(jd_text, jd_source=jd_name or os.path.basename(jd_path))
            write_ranking_file(rf, os.path.join("out", "ranking_file.yaml"))
        with st.spinner("Scrubbing PII locally (Presidio) …"):
            candidate_id = cv_name or candidate_id_from_path(cv_path)
            scrub = scrub_cv(cv_text, candidate_id)
        with st.spinner("Evaluating the scrubbed CV against the rubric (Claude) …"):
            result = evaluate_cv(rf, scrub.scrubbed_text, candidate_id)
            write_evaluation(result, "out")
        st.session_state["data"] = {"rf": rf, "scrub": scrub, "result": result}
    except ExtractionError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # surface Bedrock / parsing errors plainly
        st.error(f"{type(e).__name__}: {e}")
        st.stop()


data = st.session_state.get("data")
if not data:
    st.info("Pick a JD and a CV in the sidebar, then **Run screening**.")
    st.stop()

rf = data["rf"]
scrub = data["scrub"]
result = data["result"]
dim_by_id = {d.id: d for d in rf.dimensions}

ranking_yaml = yaml.safe_dump(rf.model_dump(), sort_keys=False, allow_unicode=True)
eval_yaml = yaml.safe_dump(result.model_dump(), sort_keys=False, allow_unicode=True)
eval_json = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

# Candidate header
st.success(f"**Candidate:** `{result.candidate_id}`  ·  **Role:** {rf.role_title}")


# --------------------------------------------------------------------------- #
# Panel 1 — Ranking file
# --------------------------------------------------------------------------- #
st.header("1 · How we'll judge")
st.caption(f"Dimensions derived from `{rf.derived_from}`. Weights sum to 1.0.")
rows = "\n".join(f"| {_cell(d.name)} | {d.weight:.0%} |" for d in rf.dimensions)
st.markdown("| Dimension | Weight |\n|:--|--:|\n" + rows)
if rf.notes.strip():
    st.caption(f"_{rf.notes.strip()}_")
with st.expander("Full ranking_file.yaml (the exact file that drives evaluation)"):
    st.code(ranking_yaml, language="yaml")
st.download_button("⬇ ranking_file.yaml", ranking_yaml, "ranking_file.yaml", "text/yaml")


# --------------------------------------------------------------------------- #
# Panel 2 — Redaction report + scrubbed preview
# --------------------------------------------------------------------------- #
st.header("2 · PII scrubbed locally")
st.caption("Removed by Presidio before any LLM call — no model ever sees the candidate's identity.")
if scrub.redaction_report:
    report_rows = "\n".join(f"| {k} | {v} |" for k, v in scrub.redaction_report.items())
    st.markdown("| Entity type | Redacted |\n|:--|--:|\n" + report_rows)
else:
    st.caption("No PII entities detected.")
with st.expander("Scrubbed CV preview (exactly what the LLM saw)"):
    st.text(scrub.scrubbed_text)


# --------------------------------------------------------------------------- #
# Panel 3 — Per-dimension scorecard (primary output)
# --------------------------------------------------------------------------- #
st.header("3 · Scorecard")
st.caption("Primary output — each dimension scored 1–5 on its own, highest-weighted first.")

ordered = sorted(
    result.dimension_scores,
    key=lambda s: dim_by_id[s.dimension_id].weight if s.dimension_id in dim_by_id else 0,
    reverse=True,
)
for s in ordered:
    dim = dim_by_id.get(s.dimension_id)
    title = dim.name if dim else s.dimension_id
    weight = dim.weight if dim else 0.0
    with st.container(border=True):
        left, right = st.columns([1, 5])
        with left:
            st.markdown(f"## {s.score}/5")
            st.caption(f"weight {weight:.0%}")
            st.progress(s.score / 5)
        with right:
            st.markdown(f"**{title}**")
            st.write(s.reasoning or "—")
            if s.evidence:
                for span, ok in zip(s.evidence, s.verified):
                    st.markdown(f"> {_cell(span)}")
                    st.markdown(
                        ":green[✓ verified in CV]" if ok
                        else ":orange[⚠ unverified — not found verbatim in the CV]"
                    )
            else:
                st.markdown(":gray[No supporting evidence quoted.]")

st.metric("Weighted composite", f"{result.weighted_composite} / 5")
st.caption(
    "Secondary context, computed in code as Σ(score × weight) — **not** the verdict. "
    "The per-dimension breakdown above is the primary output."
)


# --------------------------------------------------------------------------- #
# Panel 4 — Highlights
# --------------------------------------------------------------------------- #
st.header("4 · Highlights")
st.caption("Plain-language summary for a non-technical reader.")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### ✅ Strengths")
    for x in result.highlights.strengths:
        st.markdown(f"- {x}")
with c2:
    st.markdown("#### ⚠️ Gaps / concerns")
    for x in result.highlights.gaps_or_concerns:
        st.markdown(f"- {x}")
with c3:
    st.markdown("#### ❓ Interview probes")
    for x in result.highlights.interview_probes:
        st.markdown(f"- {x}")

st.divider()
d1, d2 = st.columns(2)
d1.download_button("⬇ evaluation YAML", eval_yaml, f"evaluation_{result.candidate_id}.yaml", "text/yaml")
d2.download_button("⬇ evaluation JSON", eval_json, f"evaluation_{result.candidate_id}.json", "application/json")
