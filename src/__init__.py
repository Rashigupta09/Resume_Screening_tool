"""Resume Screening Tool — pipeline package.

Each module is a pure function with typed I/O, reusable from app.py and cli.py:
  extract  -> PDF/text to clean text
  scrub    -> local Presidio PII redaction (+ report, candidate_id linkage)
  schema   -> Pydantic models (single source of truth for ranking + evaluation)
  llm      -> Bedrock Converse wrapper + prompts
  derive   -> Call 1: JD -> ranking_file.yaml
  evaluate -> Call 2: ranking + scrubbed CV -> scores + highlights
  verify   -> evidence span verification + weighted composite (pure Python)
"""
