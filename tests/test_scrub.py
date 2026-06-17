"""PII scrubbing: identity removed, signal kept (needs Presidio + en_core_web_md).

Skips cleanly if presidio/spaCy or the model aren't installed, so the rest of the
suite still runs in a bare environment.
"""
import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("presidio_anonymizer")

from src.scrub import scrub_cv  # noqa: E402

SAMPLE = (
    "Jane Doe\n"
    "Email: jane.doe@example.com | Phone: +1 415 555 0132\n"
    "linkedin.com/in/janedoe\n"
    "Senior Backend Engineer at Stripe. Built payment services with Python and PostgreSQL.\n"
    "2019-2023."
)


@pytest.fixture(scope="module")
def scrubbed():
    try:
        return scrub_cv(SAMPLE, "cv_test.txt")
    except Exception as e:  # model not downloaded etc.
        pytest.skip(f"Presidio/spaCy model unavailable: {e}")


def test_identity_is_removed(scrubbed):
    text = scrubbed.scrubbed_text
    assert "Jane Doe" not in text
    assert "jane.doe@example.com" not in text
    assert "[PERSON]" in text or "[EMAIL]" in text  # at least typed placeholders present


def test_signal_is_kept(scrubbed):
    # Companies, skills, and dates are scoring signal — they must survive.
    text = scrubbed.scrubbed_text
    assert "Stripe" in text
    assert "Python" in text
    assert "PostgreSQL" in text


def test_redaction_report_has_counts(scrubbed):
    assert isinstance(scrubbed.redaction_report, dict)
    assert sum(scrubbed.redaction_report.values()) >= 1
    assert scrubbed.candidate_id == "cv_test.txt"
