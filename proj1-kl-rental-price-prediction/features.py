from __future__ import annotations

import math
import re
from typing import Any


FEATURE_COLS: tuple[str, ...] = (
    "region",
    "area",
    "property_type",
    "furnished",
    "size_sqft",
    "rooms_num",
    "bathroom_num",
    "parking_num",
    "has_parking_info",
    "completion_year_num",
    "has_completion_year",
    "facility_count",
    "has_pool",
    "has_gym",
    "has_security",
    "has_sauna",
    "has_lift",
    "near_rail",
    "has_aircond",
    "has_washer",
    "cooking_allowed",
    "has_internet",
)

BINARY_AMENITY_COLS: tuple[str, ...] = (
    "has_pool",
    "has_gym",
    "has_security",
    "has_sauna",
    "has_lift",
    "near_rail",
    "has_aircond",
    "has_washer",
    "cooking_allowed",
    "has_internet",
)

KNOWN_REGIONS: frozenset[str] = frozenset({"kuala lumpur", "selangor"})

KNOWN_PROPERTY_TYPES: frozenset[str] = frozenset(
    {
        "condominium",
        "apartment",
        "service residence",
        "flat",
        "studio",
        "duplex",
        "townhouse condo",
        "others",
    }
)

KNOWN_FURNISHED_VALUES: frozenset[str] = frozenset(
    {
        "fully furnished",
        "partially furnished",
        "not furnished",
    }
)


def _is_missing(val: Any) -> bool:
    return val is None or (isinstance(val, float) and math.isnan(val))


def extract_number(val: Any) -> float:
    """Extract the first integer or decimal token from a raw string."""
    if _is_missing(val):
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", str(val))
    return float(match.group(1)) if match else float("nan")


def parse_rooms(val: Any) -> float:
    """Parse room strings such as '3', '3.0', or 'More than 10'."""
    if _is_missing(val):
        return float("nan")
    text = str(val).strip().lower()
    if "more than 10" in text:
        return 11.0
    return extract_number(text)


def canonical_tokens(val: Any, drop_numeric: bool = True) -> str:
    """Canonicalize comma-separated facility tags for duplicate detection."""
    if _is_missing(val):
        return ""
    tokens = [t.strip().lower() for t in str(val).split(",") if t.strip()]
    if drop_numeric:
        tokens = [t for t in tokens if not t.isdigit()]
    return "|".join(sorted(set(tokens)))


def count_valid_facilities(val: Any) -> int:
    """Count non-numeric comma-separated facility tokens."""
    if _is_missing(val):
        return 0
    return len(
        [t for t in str(val).split(",") if t.strip() and not t.strip().isdigit()]
    )


def normalize_region(val: Any) -> str:
    if _is_missing(val):
        return "unknown"
    cleaned = str(val).strip().lower()
    return cleaned or "unknown"


def normalize_area(val: Any, top_areas: set[str] | frozenset[str] | None = None) -> str:
    """Normalize township area name and map rare or numeric IDs to 'other'."""
    if _is_missing(val):
        return "other"
    text = str(val).strip().lower()
    if " - " in text:
        text = text.split(" - ", 1)[1].strip()
    if not text or text.isdigit():
        return "other"
    if top_areas is not None and text not in top_areas:
        return "other"
    return text


def normalize_property_type(val: Any) -> str:
    if _is_missing(val):
        return "others"
    cleaned = str(val).strip().lower()
    return cleaned if cleaned in KNOWN_PROPERTY_TYPES else "others"


def normalize_furnished(val: Any) -> str:
    if _is_missing(val):
        return "not furnished"
    cleaned = str(val).strip().lower()
    if not cleaned or cleaned == "unknown":
        return "not furnished"
    return cleaned


def prepare_inference_record(
    payload: dict[str, Any], prep_artifacts: dict[str, Any]
) -> dict[str, Any]:
    """Convert one API JSON payload into the exact 22-feature dict expected by DictVectorizer."""
    top_areas = set(prep_artifacts["top_areas"])
    year_median = float(prep_artifacts["year_median"])

    parking_val = payload.get("parking_num")
    if _is_missing(parking_val):
        has_parking_info = 0
        parking_num = 0.0
    else:
        raw_has_parking = payload.get("has_parking_info")
        has_parking_info = 1 if raw_has_parking is None else int(bool(raw_has_parking))
        parking_num = float(parking_val) if has_parking_info else 0.0

    year_val = payload.get("completion_year_num")
    if _is_missing(year_val):
        has_completion_year = 0
        completion_year_num = year_median
    else:
        raw_has_year = payload.get("has_completion_year")
        has_completion_year = 1 if raw_has_year is None else int(bool(raw_has_year))
        completion_year_num = float(year_val) if has_completion_year else year_median

    record: dict[str, Any] = {
        "region": normalize_region(payload.get("region")),
        "area": normalize_area(payload.get("area"), top_areas=top_areas),
        "property_type": normalize_property_type(payload.get("property_type")),
        "furnished": normalize_furnished(payload.get("furnished")),
        "size_sqft": float(payload["size_sqft"]),
        "rooms_num": float(payload["rooms_num"]),
        "bathroom_num": float(payload["bathroom_num"]),
        "parking_num": parking_num,
        "has_parking_info": has_parking_info,
        "completion_year_num": completion_year_num,
        "has_completion_year": has_completion_year,
        "facility_count": int(payload.get("facility_count") or 0),
    }
    for col in BINARY_AMENITY_COLS:
        record[col] = int(bool(payload.get(col, 0)))

    return record
