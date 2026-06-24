from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")
EXPAND_COLLAPSE_RE = re.compile(r"(?:\s*show\s+(?:more|less)\s*)+$", re.IGNORECASE)

EXPECTED_COLLECTIONS = (
    "skills",
    "experience",
    "education",
    "awards",
    "courses",
    "certifications",
    "publications",
    "patents",
)

PROFILE_YEAR_FIELDS = {
    "date_from_year",
    "date_to_year",
    "begin_year",
    "end_year",
}
MONTH_FIELDS = {"date_from_month", "date_to_month"}
NON_NEGATIVE_FIELDS = {
    "duration_months",
    "total_experience_duration_months",
    "company_employees_count",
    "order_in_profile",
    "degree_level",
}
SENTINEL_MINUS_ONE_FIELDS = {"company_size_range"}


def normalize_text(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value)
    text = ZERO_WIDTH_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = EXPAND_COLLAPSE_RE.sub("", text).strip()
    return text or None


def _dedupe_list(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        marker = repr(value).casefold() if isinstance(value, (str, int, float, bool)) else repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _clean_value(key: str | None, value: Any, warnings: list[str], path: str) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        cleaned = [
            _clean_value(None, item, warnings, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return _dedupe_list(cleaned)
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for child_key, child_value in value.items():
            cleaned = _clean_value(
                child_key,
                child_value,
                warnings,
                f"{path}.{child_key}" if path else child_key,
            )
            if cleaned is not None and cleaned != {}:
                output[child_key] = cleaned
        return output
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        current_year = datetime.now(UTC).year
        if key in SENTINEL_MINUS_ONE_FIELDS and value == -1:
            return None
        if key in PROFILE_YEAR_FIELDS and not (1900 <= value <= current_year + 1):
            warnings.append(f"{path}: invalid year {value}")
            return None
        if key == "company_founded_year" and not (1500 <= value <= current_year + 1):
            warnings.append(f"{path}: invalid company founded year {value}")
            return None
        if key in MONTH_FIELDS and not (1 <= value <= 12):
            warnings.append(f"{path}: invalid month {value}")
            return None
        if key in NON_NEGATIVE_FIELDS and value < 0:
            warnings.append(f"{path}: negative value {value}")
            return None
    return value


def _validate_chronology(profile: dict[str, Any], warnings: list[str]) -> None:
    for index, item in enumerate(profile.get("experience", [])):
        start = item.get("start_time")
        end = item.get("end_time")
        if start and end and end < start:
            warnings.append(f"experience[{index}]: end_time precedes start_time")
    for index, item in enumerate(profile.get("education", [])):
        begin = item.get("begin_year")
        end = item.get("end_year")
        if begin and end and end < begin:
            warnings.append(f"education[{index}]: end_year precedes begin_year")


def clean_profile(raw_profile: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize one profile while retaining meaningful missingness."""
    warnings: list[str] = []
    profile = _clean_value(None, deepcopy(raw_profile), warnings, "")
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a JSON object")

    for field in EXPECTED_COLLECTIONS:
        value = profile.get(field)
        if value is None:
            profile[field] = []
        elif not isinstance(value, list):
            warnings.append(f"{field}: expected list, got {type(value).__name__}")
            profile[field] = [value]

    if not isinstance(profile.get("user_id"), (str, int)):
        raise ValueError("Profile has no valid user_id")

    profile["skills"] = _dedupe_list(
        [skill for skill in profile["skills"] if isinstance(skill, str) and skill]
    )
    profile["experience"] = sorted(
        [item for item in profile["experience"] if isinstance(item, dict)],
        key=lambda item: item.get("order_in_profile", 10**9),
    )
    profile["education"] = sorted(
        [item for item in profile["education"] if isinstance(item, dict)],
        key=lambda item: item.get("order_in_profile", 10**9),
    )
    _validate_chronology(profile, warnings)
    return profile, warnings
