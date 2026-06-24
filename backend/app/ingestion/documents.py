from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.documents import Document


def _unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _current_experience(profile: dict[str, Any]) -> dict[str, Any]:
    return next(
        (item for item in profile["experience"] if item.get("is_current") is True),
        profile["experience"][0] if profile["experience"] else {},
    )


def build_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    current = _current_experience(profile)
    experience = profile["experience"]
    education = profile["education"]
    highest_degree = max(
        (item.get("degree_level") for item in education if isinstance(item.get("degree_level"), int)),
        default=None,
    )
    total_months = profile.get("total_experience_duration_months")
    return {
        "candidate_id": str(profile["user_id"]),
        "headline": profile.get("headline"),
        "is_working": profile.get("is_working"),
        "current_title": profile.get("active_experience_title") or current.get("title"),
        "current_department": profile.get("active_experience_department") or current.get("role"),
        "management_level": profile.get("active_experience_management_level")
        or current.get("level"),
        "is_decision_maker": profile.get("is_decision_maker"),
        "total_experience_months": total_months,
        "total_experience_years": round(total_months / 12, 1)
        if isinstance(total_months, (int, float))
        else None,
        "highest_degree_level": highest_degree,
        "skills": _unique_strings(profile["skills"]),
        "roles": _unique_strings(item.get("role") for item in experience),
        "levels": _unique_strings(item.get("level") for item in experience),
        "industries": _unique_strings(item.get("industry") for item in experience),
        "companies": _unique_strings(item.get("company_name") for item in experience),
        "locations": _unique_strings(
            value
            for item in experience
            for value in (
                item.get("address_city"),
                item.get("address_state"),
                item.get("address_country"),
            )
        ),
        "majors": _unique_strings(item.get("major") for item in education),
        "education_countries": _unique_strings(
            item.get("institution_country") for item in education
        ),
    }


def _format_named_items(title: str, items: list[Any]) -> str | None:
    rendered: list[str] = []
    for item in items:
        if isinstance(item, str):
            rendered.append(item)
        elif isinstance(item, dict):
            values = [
                str(value)
                for key, value in item.items()
                if value not in (None, "", [], {}) and key not in {"order_in_profile"}
            ]
            if values:
                rendered.append(" | ".join(values))
    return f"{title}: " + "; ".join(rendered) if rendered else None


def build_page_content(profile: dict[str, Any]) -> str:
    sections: list[str | None] = [
        f"Headline: {profile.get('headline')}" if profile.get("headline") else None,
        f"Summary: {profile.get('summary')}" if profile.get("summary") else None,
        "Skills: " + ", ".join(profile["skills"]) if profile["skills"] else None,
    ]

    current = _current_experience(profile)
    if current:
        current_bits = [
            current.get("title"),
            current.get("company_name"),
            current.get("role"),
            current.get("level"),
        ]
        sections.append("Current/Latest role: " + " | ".join(bit for bit in current_bits if bit))

    if profile["experience"]:
        experience_lines: list[str] = []
        for item in profile["experience"]:
            period = " - ".join(
                value
                for value in (
                    item.get("start_time"),
                    item.get("end_time") or ("present" if item.get("is_current") else None),
                )
                if value
            )
            core = " | ".join(
                value
                for value in (
                    item.get("title"),
                    item.get("company_name"),
                    item.get("role"),
                    item.get("level"),
                    item.get("industry"),
                    period or None,
                )
                if value
            )
            description = item.get("description")
            experience_lines.append(f"- {core}" + (f". {description}" if description else ""))
        sections.append("Experience:\n" + "\n".join(experience_lines))

    if profile["education"]:
        education_lines = []
        for item in profile["education"]:
            years = " - ".join(
                str(value)
                for value in (item.get("begin_year"), item.get("end_year"))
                if value is not None
            )
            education_lines.append(
                "- "
                + " | ".join(
                    str(value)
                    for value in (
                        item.get("degree_str"),
                        item.get("major"),
                        item.get("institution_name"),
                        years or None,
                    )
                    if value
                )
            )
        sections.append("Education:\n" + "\n".join(education_lines))

    for field, title in (
        ("certifications", "Certifications"),
        ("courses", "Courses"),
        ("awards", "Awards"),
        ("publications", "Publications"),
        ("patents", "Patents"),
    ):
        sections.append(_format_named_items(title, profile[field]))

    return "\n\n".join(section for section in sections if section)


def profile_to_document(profile: dict[str, Any]) -> Document:
    return Document(page_content=build_page_content(profile), metadata=build_metadata(profile))


def metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)
