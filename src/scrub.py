"""Local PII scrubbing with Microsoft Presidio (spaCy en_core_web_md backend).

NO LLM is used here — the brief is explicit that PII removal must be local. We
scrub identity (names, contact details, addresses, personal profile URLs) and
deliberately KEEP signal (companies, job titles, skills, dates, education) so
scoring quality is preserved. Over-scrubbing tanks scoring quality.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

# Entities we remove. Deliberately EXCLUDES ORG and DATE_TIME so company names
# and dates survive as scoring signal. LOCATION covers physical addresses (it can
# also catch city names — an accepted trade-off; tune this list to change it).
PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "URL"]

PLACEHOLDERS = {
    "PERSON": "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "LOCATION": "[ADDRESS]",
    "URL": "[URL]",
}

# Drop weak detections. The URL recognizer fires at ~0.5 on fragments like
# "B.Tech" or "gmail.com" (education / email fragments — signal, not PII), while
# real names/emails/phones/profile URLs score >= 0.75.
SCORE_THRESHOLD = 0.6

# spaCy (en_core_web_md) sometimes tags tools/skills as PERSON. These are scoring
# signal and must never be scrubbed — keep them regardless of the NER label.
SKILL_ALLOWLIST = {
    "python", "java", "javascript", "typescript", "go", "golang", "rust", "scala",
    "django", "flask", "fastapi", "spring", "react", "node", "node.js", "express",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "sqlite", "elasticsearch",
    "kafka", "rabbitmq", "sqs", "sns", "s3", "ec2", "rds", "lambda", "dynamodb",
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform", "ansible",
    "jenkins", "git", "github", "gitlab", "ci", "cd", "ci/cd", "rest", "grpc",
    "graphql", "celery", "pytest", "nginx", "linux", "bash", "html", "css", "sql",
    "etl", "ml", "nlp", "llm", "rag", "spark", "airflow", "kibana", "grafana",
}

_analyzer = None
_anonymizer = None


@dataclass
class ScrubResult:
    candidate_id: str
    scrubbed_text: str
    redaction_report: Dict[str, int] = field(default_factory=dict)


def candidate_id_from_path(cv_path: str) -> str:
    """The non-PII linkage back to the source file (e.g. 'cv_03.pdf')."""
    return os.path.basename(cv_path)


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
            }
        )
        nlp_engine = provider.create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        # Default recognizers already cover EMAIL_ADDRESS / PHONE_NUMBER / URL via regex,
        # so LinkedIn / personal GitHub links are scrubbed as URLs.
    if _anonymizer is None:
        from presidio_anonymizer import AnonymizerEngine

        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def scrub_cv(text: str, candidate_id: str) -> ScrubResult:
    """Replace PII with typed placeholders; return scrubbed text + a count report."""
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(
        text=text, entities=PII_ENTITIES, language="en", score_threshold=SCORE_THRESHOLD
    )
    # Never scrub known tools/skills even if NER mislabels them as a person.
    results = [
        r for r in results
        if text[r.start:r.end].strip().lower() not in SKILL_ALLOWLIST
    ]

    operators = {
        entity: OperatorConfig("replace", {"new_value": placeholder})
        for entity, placeholder in PLACEHOLDERS.items()
    }
    anonymized = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    )

    # Count actually-applied replacements (post overlap-resolution) per entity type,
    # so the coordinator can trust that scrubbing happened.
    report: Dict[str, int] = {}
    for item in anonymized.items:
        report[item.entity_type] = report.get(item.entity_type, 0) + 1

    return ScrubResult(
        candidate_id=candidate_id,
        scrubbed_text=anonymized.text,
        redaction_report=dict(sorted(report.items())),
    )
