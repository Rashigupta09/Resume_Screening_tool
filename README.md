# Resume Screening Tool (Proof of Concept)

Screen **one CV against one job description** and get a transparent, multi-dimensional,
evidence-backed fit assessment a non-technical HR coordinator can act on. The value is
**explainability, not speed**: every score traces to a dimension derived from the JD, and
every quoted piece of evidence is verified against the candidate's own text.

## What I built

An end-to-end pipeline (CLI + Streamlit UI) that:

1. **Accepts** one JD + one CV as PDF or plain text.
2. **Scrubs PII from the CV locally** with Presidio + spaCy — **no LLM** — before any model
   call. Identity is removed (name, email, phone, address, personal profile URLs); signal is
   kept (companies, titles, skills, dates). Results link back to the candidate by **filename**.
3. **Derives ranking dimensions from the JD** (Claude Sonnet 4.6, "Call 1"): 4–7 weighted
   dimensions inferred from the JD itself — nothing hardcoded — written to an inspectable
   `out/ranking_file.yaml`. That same file is read back to drive scoring (one source of truth).
4. **Evaluates the CV** against that rubric (Claude, "Call 2"): every dimension scored 1–5,
   each with a one-line rationale and **verbatim evidence quotes** from the CV.
5. **Verifies + scores in Python**: each evidence span is checked by normalized substring
   match against the scrubbed CV (unverified spans are flagged, not hidden), and the weighted
   composite is computed **in code, not by the LLM**.
6. **Summarizes** strengths / gaps / interview probes in plain language, and renders
   everything in a 4-panel UI with downloadable YAML/JSON artifacts.

**Differentiator — evidence-linked, verified scoring:** scores quote the exact CV span that
justifies them, and those spans are verified against the source — transparency plus a
hallucination guard. **Judgment vs. arithmetic are separated:** the LLM scores; Python does
the weighting and verification.

## How to run locally

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate                 # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_md

# 2. Config — copy the template and add your Bedrock API key
copy .env.example .env                  # macOS/Linux: cp .env.example .env
#   then set AWS_BEARER_TOKEN_BEDROCK=ABSK...  (your Bedrock API key) in .env

# 3. Run — CLI ...
python cli.py --jd data/job_description.txt --cv data/cv_01.txt
# 3. ... or the UI
streamlit run app.py

# Tests
pytest
```

Outputs are written to `out/`: `ranking_file.yaml` and `evaluation_<candidate_id>.yaml`/`.json`.
A sample JD + CV are included in `data/`.

## Non-local dependencies

- **AWS Bedrock** with **Claude Sonnet 4.6 access enabled in `us-east-1`**.
- A **Bedrock API key** (bearer token, starts with `ABSK`), set as `AWS_BEARER_TOKEN_BEDROCK`.
  Auth is via this key — not IAM/SigV4. Model ID: `us.anthropic.claude-sonnet-4-6` (the bare
  `anthropic.claude-sonnet-4-6` is rejected for on-demand use and needs this inference profile).
- Optional: set `EVAL_MODEL_ID=us.anthropic.claude-haiku-4-5` to speed up Call 2 only.

Everything else (PII scrubbing, schema validation, evidence verification, weighting) runs
locally with no network.

## Key assumptions

- **Text-based PDFs only** — scanned/image PDFs yield no text and raise a clear error (no OCR).
- **One JD + one CV per run** (multi-CV is a natural extension, not built).
- **The JD is not scrubbed** — it's a hiring-manager document; only CVs are scrubbed.
- **1–5 score per dimension**; the weighted total is computed in code and shown only as
  secondary, clearly-derived context — never as "the" verdict.
- **Evidence is verified** by normalized substring match against the scrubbed CV; unverified
  spans are flagged, not hidden.
- PII scrubbing is tuned to protect signal (confidence threshold + a skills/tools allow-list).
  `en_core_web_md` may still over-scrub an unusual **company name** — this is visible in the
  redaction report; `en_core_web_lg` reduces it at the cost of size/speed.
