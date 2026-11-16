from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import requests

from features import prepare_inference_record


MODEL_PATH = Path(__file__).with_name("model.bin")

SAMPLE_KL_CONDO: dict[str, Any] = {
    "region": "Kuala Lumpur",
    "area": "Mont Kiara",
    "property_type": "Condominium",
    "furnished": "Fully Furnished",
    "size_sqft": 1200.0,
    "rooms_num": 3,
    "bathroom_num": 2,
    "parking_num": 2,
    "completion_year_num": 2019,
    "facility_count": 9,
    "has_pool": 1,
    "has_gym": 1,
    "has_security": 1,
    "has_sauna": 1,
    "has_lift": 1,
    "near_rail": 0,
    "has_aircond": 1,
    "has_washer": 1,
    "cooking_allowed": 1,
    "has_internet": 1,
}

SAMPLE_SELANGOR_APARTMENT: dict[str, Any] = {
    "region": "selangor",
    "area": "unknown-township-xyz",
    "property_type": "apartment",
    "furnished": "not furnished",
    "size_sqft": 800.0,
    "rooms_num": 3,
    "bathroom_num": 2,
    "parking_num": None,
    "completion_year_num": None,
    "facility_count": 2,
    "has_security": 1,
    "has_lift": 1,
}

INVALID_CASES: list[tuple[str, dict[str, Any]]] = [
    ("invalid region", {**SAMPLE_KL_CONDO, "region": "penang"}),
    ("typo furnished", {**SAMPLE_KL_CONDO, "furnished": "fully-furnished"}),
    ("invalid property_type", {**SAMPLE_KL_CONDO, "property_type": "bungalow"}),
    ("rooms_num=0", {**SAMPLE_KL_CONDO, "rooms_num": 0}),
    ("bathroom_num=0", {**SAMPLE_KL_CONDO, "bathroom_num": 0}),
    ("facility_count=15", {**SAMPLE_KL_CONDO, "facility_count": 15}),
]


def check_schema_validation() -> None:
    """Verify Pydantic schema accepts case-insensitive valid values and rejects invalid ones."""
    from pydantic import ValidationError
    from predict import RentalListing

    valid = RentalListing(
        **{
            **SAMPLE_KL_CONDO,
            "region": "  Kuala Lumpur ",
            "furnished": "FULLY FURNISHED",
            "property_type": "Service Residence",
        }
    )
    assert valid.region == "kuala lumpur"
    assert valid.furnished == "fully furnished"
    assert valid.property_type == "service residence"

    for label, bad_payload in INVALID_CASES:
        try:
            RentalListing(**bad_payload)
        except ValidationError:
            continue
        raise AssertionError(f"Expected ValidationError for {label}")
    print(f"Schema validation OK ({len(INVALID_CASES)} invalid cases rejected)")


def check_offline_artifact() -> None:
    """Verify saved model.bin loads cleanly and gives deterministic predictions."""
    check_schema_validation()
    with MODEL_PATH.open("rb") as f:
        artifact = pickle.load(f)

    prep = artifact["prep_artifacts"]
    dv = artifact["dv"]
    model = artifact["model"]

    kl_rec = prepare_inference_record(SAMPLE_KL_CONDO, prep)
    sel_rec = prepare_inference_record(SAMPLE_SELANGOR_APARTMENT, prep)

    assert kl_rec["area"] == "mont kiara"
    assert kl_rec["has_completion_year"] == 1
    assert sel_rec["area"] == "other"
    assert sel_rec["has_completion_year"] == 0
    assert sel_rec["completion_year_num"] == prep["year_median"]
    assert sel_rec["has_parking_info"] == 0

    pred_kl_1 = float(np.expm1(model.predict(dv.transform([kl_rec]))[0]))
    pred_kl_2 = float(np.expm1(model.predict(dv.transform([kl_rec]))[0]))
    pred_sel = float(np.expm1(model.predict(dv.transform([sel_rec]))[0]))

    assert pred_kl_1 == pred_kl_2
    assert 500 < pred_sel < pred_kl_1 < 15000
    print(
        f"Offline parity OK | KL Mont Kiara condo: RM {pred_kl_1:,.2f}/mo | "
        f"Selangor apartment (imputed): RM {pred_sel:,.2f}/mo"
    )


def check_http_service(base_url: str) -> None:
    """Verify live FastAPI /health and /predict endpoints."""
    health_resp = requests.get(f"{base_url}/health", timeout=60)
    health_resp.raise_for_status()
    health_data = health_resp.json()
    assert health_data["status"] == "ok"

    resp_kl = requests.post(f"{base_url}/predict", json=SAMPLE_KL_CONDO, timeout=60)
    resp_kl.raise_for_status()
    data_kl = resp_kl.json()

    resp_sel = requests.post(f"{base_url}/predict", json=SAMPLE_SELANGOR_APARTMENT, timeout=60)
    resp_sel.raise_for_status()
    data_sel = resp_sel.json()

    assert data_kl["normalized_area"] == "mont kiara"
    assert data_kl["completion_year_imputed"] is False
    assert data_sel["normalized_area"] == "other"
    assert data_sel["completion_year_imputed"] is True
    assert data_kl["predicted_monthly_rent_myr"] > data_sel["predicted_monthly_rent_myr"]

    for label, bad_payload in INVALID_CASES:
        bad_resp = requests.post(f"{base_url}/predict", json=bad_payload, timeout=30)
        assert bad_resp.status_code == 422, f"{label} expected 422, got {bad_resp.status_code}"

    print(f"GET /health OK -> {health_data}")
    print(f"POST /predict (KL condo) -> {data_kl}")
    print(f"POST /predict (Selangor apt) -> {data_sel}")
    print(f"Strict 422 validation checks passed ({len(INVALID_CASES)} cases)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for KL rental price prediction.")
    parser.add_argument(
        "--url",
        default=os.getenv("API_URL", "http://127.0.0.1:9696"),
        help="Base URL of running prediction service",
    )
    parser.add_argument("--offline-only", action="store_true", help="Only run local artifact check")
    parser.add_argument("--http-only", action="store_true", help="Only run HTTP endpoint checks")
    args = parser.parse_args()

    if not args.http_only:
        check_offline_artifact()
    if not args.offline_only:
        check_http_service(args.url.rstrip("/"))


if __name__ == "__main__":
    main()
