"""Call 1: derive ranking dimensions from the JD and write ranking_file.yaml.

The written file is the single source of truth — evaluate.py reads it back.
Dimensions are derived from the JD itself; nothing here is hardcoded.
"""
from __future__ import annotations

import os

import yaml

from .llm import DERIVE_SYSTEM, DERIVE_USER_TEMPLATE, call_claude_json
from .schema import RankingFile

DEFAULT_RANKING_PATH = os.path.join("out", "ranking_file.yaml")


def derive_dimensions(jd_text: str, jd_source: str = "") -> RankingFile:
    data = call_claude_json(
        DERIVE_SYSTEM,
        DERIVE_USER_TEMPLATE.format(jd_text=jd_text),
        max_tokens=4096,
    )

    # Be robust to small omissions before validation.
    for i, d in enumerate(data.get("dimensions", []), start=1):
        d.setdefault("id", f"dim_{i}")
        d.setdefault("what_to_look_for", "")
        d.setdefault("positive_signals", [])
        d.setdefault("negative_signals", [])
    if jd_source:
        data["derived_from"] = jd_source

    return RankingFile.model_validate(data)


def write_ranking_file(rf: RankingFile, path: str = DEFAULT_RANKING_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(rf.model_dump(), fh, sort_keys=False, allow_unicode=True)
    return path


def load_ranking_file(path: str = DEFAULT_RANKING_PATH) -> RankingFile:
    """Read the SAME file back as the rubric that drives evaluation."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return RankingFile.model_validate(data)
