"""Bedrock Converse wrapper + paste-ready prompts (PRD section 9).

Both reasoning calls use Claude Sonnet 4.6 via bedrock-runtime.converse. JSON is
requested by prompt and validated downstream with Pydantic (judgment from the
LLM; arithmetic and verification stay in Python).
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Confirmed-working us-east-1 cross-region inference profile for Sonnet 4.6.
# The bare foundation-model id (anthropic.claude-sonnet-4-6) is rejected for
# on-demand throughput — it requires this inference profile.
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-6")

# Call 2 (evaluation) may use a faster/cheaper model than Call 1 (derivation).
# Defaults to MODEL_ID; set EVAL_MODEL_ID=us.anthropic.claude-haiku-4-5 to speed up
# scoring (the PRD's documented Haiku-for-Call-2 option) if you have Haiku access.
EVAL_MODEL_ID = os.getenv("EVAL_MODEL_ID", MODEL_ID)

# Auth is a Bedrock API key (bearer token, "ABSK..."), NOT IAM/SigV4. botocore
# reads it from AWS_BEARER_TOKEN_BEDROCK automatically; load_dotenv() above puts
# a .env value into the environment so the boto3 client picks it up.
BEDROCK_API_KEY_ENV = "AWS_BEARER_TOKEN_BEDROCK"

_client = None


def _bedrock():
    global _client
    if _client is None:
        if not os.getenv(BEDROCK_API_KEY_ENV):
            raise RuntimeError(
                f"{BEDROCK_API_KEY_ENV} is not set. Put your Bedrock API key (ABSK...) "
                f"in .env as {BEDROCK_API_KEY_ENV}=... (see .env.example). "
                "Auth is via the API key, not IAM credentials."
            )
        import boto3  # lazy: pure-Python tests don't need AWS

        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def call_claude(
    system: str, user: str, max_tokens: int = 4096, temperature: float = 0.0,
    model_id: str | None = None,
) -> str:
    """One Bedrock Converse turn; returns the assistant's text block."""
    resp = _bedrock().converse(
        modelId=model_id or MODEL_ID,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return resp["output"]["message"]["content"][0]["text"]


def _strip_fences(text: str) -> str:
    """Drop a leading ```json / ``` and trailing ``` if the model wrapped output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    candidate = _strip_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)  # first {...} block
        if match:
            return json.loads(match.group(0))
        raise


def call_claude_json(
    system: str, user: str, max_tokens: int = 4096, model_id: str | None = None
) -> dict:
    """call_claude + parse_json, with one retry that nudges 'JSON only'."""
    raw = call_claude(system, user, max_tokens=max_tokens, model_id=model_id)
    try:
        return parse_json(raw)
    except json.JSONDecodeError:
        retry_user = user + "\n\nReturn ONLY valid JSON. No prose, no markdown fences."
        raw = call_claude(system, retry_user, max_tokens=max_tokens, model_id=model_id)
        return parse_json(raw)


# --------------------------------------------------------------------------- #
# Prompts (PRD section 9) — content is never hardcoded; only the shape is.
# --------------------------------------------------------------------------- #
DERIVE_SYSTEM = (
    "You are an expert technical recruiter. Given a job description, derive the criteria a "
    "strong reviewer would use to evaluate candidates. Derive everything from the JD itself - "
    "do not use generic or hardcoded criteria. Output strict JSON only, no prose."
)

DERIVE_USER_TEMPLATE = """JOB DESCRIPTION:
{jd_text}

Produce 4-7 evaluation dimensions. For each: a short name; a weight (0-1, reflecting how
much the JD emphasizes it via ordering, repetition, and must-have vs nice-to-have
language - weights must sum to 1.0); what_to_look_for (concrete, role-specific);
positive_signals (list); negative_signals (list).

Return JSON only:
{{
  "role_title": "...",
  "dimensions": [
    {{"id":"dim_1","name":"...","weight":0.0,"what_to_look_for":"...",
     "positive_signals":["..."],"negative_signals":["..."]}}
  ],
  "scoring_scale": "1-5 (1=no evidence, 5=strong direct evidence)"
}}"""

EVALUATE_SYSTEM = (
    "You are an expert technical recruiter evaluating one candidate against a fixed rubric. "
    "Score strictly on evidence present in the CV. For each dimension, quote the exact span "
    "from the CV that justifies the score - copy it verbatim, do not paraphrase. If there is "
    "no supporting evidence, score low and say so. Write the summary for a non-technical HR "
    "reader. Output strict JSON only."
)

EVALUATE_USER_TEMPLATE = """RUBRIC (derived from the JD):
{ranking_json}

CANDIDATE CV (PII already removed; refer to the candidate as "the candidate"):
{scrubbed_cv}

For each dimension return: score (1-5), evidence (1-2 SHORT verbatim CV spans copied
exactly, empty list if none), reasoning (ONE concise sentence, max ~25 words). Then a
highlights object: strengths, gaps_or_concerns, interview_probes (specific questions a
reviewer should ask) - 3 to 4 items each, one short sentence each, plain language for a
non-technical reader. Be concise; do not pad.

Return JSON only:
{{
  "dimensions":[{{"id":"dim_1","score":0,"evidence":["..."],"reasoning":"..."}}],
  "highlights":{{"strengths":["..."],"gaps_or_concerns":["..."],"interview_probes":["..."]}}
}}

Do NOT compute an overall score - that is computed separately."""
