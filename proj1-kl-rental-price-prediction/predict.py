from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from features import (
    KNOWN_FURNISHED_VALUES,
    KNOWN_PROPERTY_TYPES,
    KNOWN_REGIONS,
    prepare_inference_record,
)


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.bin")))
METADATA_PATH = Path(os.getenv("METADATA_PATH", Path(__file__).with_name("model_metadata.json")))

with MODEL_PATH.open("rb") as model_file:
    artifact = pickle.load(model_file)

PREP_ARTIFACTS: dict[str, Any] = artifact["prep_artifacts"]
DV = artifact["dv"]
MODEL = artifact["model"]
MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

METADATA: dict[str, Any] = (
    json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if METADATA_PATH.exists()
    else {}
)

app = FastAPI(
    title="KL & Selangor Rental Price Prediction API",
    description="Predicts monthly apartment rent (MYR) in Kuala Lumpur & Selangor using a tuned XGBoost pipeline.",
    version="1.0.0",
)


def _validate_choice(value: str, allowed: frozenset[str], field_name: str) -> str:
    """Validate case-insensitive categorical membership so unseen typos do not silently zero out."""
    normalized = value.strip().lower()
    if normalized not in allowed:
        valid_str = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {field_name} '{value}'. Must be one of: {valid_str}")
    return normalized


class RentalListing(BaseModel):
    region: str = Field(..., examples=["kuala lumpur"])
    area: str = Field(..., min_length=1, examples=["mont kiara"])
    property_type: str = Field(default="condominium", examples=["condominium"])
    furnished: str = Field(default="partially furnished", examples=["fully furnished"])
    size_sqft: float = Field(..., ge=250, le=5000, examples=[1150.0])
    rooms_num: float = Field(..., ge=1, le=6, examples=[3.0])
    bathroom_num: float = Field(..., ge=1, le=6, examples=[2.0])
    parking_num: float | None = Field(default=None, ge=0, le=4, examples=[2.0])
    completion_year_num: float | None = Field(default=None, ge=1960, le=2035, examples=[2019.0])
    facility_count: int = Field(default=0, ge=0, le=14, examples=[8])
    has_pool: int = Field(default=0, ge=0, le=1)
    has_gym: int = Field(default=0, ge=0, le=1)
    has_security: int = Field(default=0, ge=0, le=1)
    has_sauna: int = Field(default=0, ge=0, le=1)
    has_lift: int = Field(default=0, ge=0, le=1)
    near_rail: int = Field(default=0, ge=0, le=1)
    has_aircond: int = Field(default=0, ge=0, le=1)
    has_washer: int = Field(default=0, ge=0, le=1)
    cooking_allowed: int = Field(default=0, ge=0, le=1)
    has_internet: int = Field(default=0, ge=0, le=1)

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        return _validate_choice(v, KNOWN_REGIONS, "region")

    @field_validator("property_type")
    @classmethod
    def validate_property_type(cls, v: str) -> str:
        return _validate_choice(v, KNOWN_PROPERTY_TYPES, "property_type")

    @field_validator("furnished")
    @classmethod
    def validate_furnished(cls, v: str) -> str:
        return _validate_choice(v, KNOWN_FURNISHED_VALUES, "furnished")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_type": METADATA.get("model_type", "XGBRegressor"),
        "model_sha256": MODEL_SHA256,
        "top_areas_count": len(PREP_ARTIFACTS["top_areas_list"]),
        "completion_year_median": PREP_ARTIFACTS["year_median"],
        "holdout_test_metrics": METADATA.get("holdout_test_metrics"),
    }


@app.post("/predict")
def predict(listing: RentalListing) -> dict[str, Any]:
    payload = listing.model_dump()
    record = prepare_inference_record(payload, PREP_ARTIFACTS)
    features_matrix = DV.transform([record])

    pred_log = float(MODEL.predict(features_matrix)[0])
    pred_myr = float(max(0.0, np.expm1(pred_log)))

    return {
        "predicted_monthly_rent_myr": round(pred_myr, 2),
        "predicted_log_rent": round(pred_log, 6),
        "normalized_area": record["area"],
        "completion_year_imputed": listing.completion_year_num is None,
        "model_sha256": MODEL_SHA256[:12],
    }
