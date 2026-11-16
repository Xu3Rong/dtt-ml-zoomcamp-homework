from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from features import (
    FEATURE_COLS,
    KNOWN_PROPERTY_TYPES,
    canonical_tokens,
    count_valid_facilities,
    extract_number,
    parse_rooms,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "mudah-apartment-kl-selangor.csv"
MODEL_PATH = PROJECT_DIR / "model.bin"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"

TOP_N_AREAS = 50
RANDOM_STATE = 42
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 2024,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def prepare_data(df_input: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Mudah.my rental listings and engineer the 22 model features."""
    df = df_input.copy()

    # 1. Deduplicate by ads_id, then canonicalized raw non-ID attributes
    if "ads_id" in df.columns:
        df = df.drop_duplicates(subset=["ads_id"], keep="first").copy()

    df["_fac_canon"] = df["facilities"].apply(
        lambda x: canonical_tokens(x, drop_numeric=True)
    )
    df["_add_canon"] = df["additional_facilities"].apply(
        lambda x: canonical_tokens(x, drop_numeric=False)
    )
    dedup_cols = [
        c for c in df.columns if c not in ["ads_id", "facilities", "additional_facilities"]
    ]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()

    # 2. Target & physical sanity bounds
    df["monthly_rent"] = pd.to_numeric(
        df["monthly_rent"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .apply(extract_number),
        errors="coerce",
    )
    df["size_sqft"] = df["size"].apply(extract_number)
    df["psf"] = df["monthly_rent"] / df["size_sqft"]

    df = df[
        df["monthly_rent"].between(300, 25000)
        & df["size_sqft"].between(250, 5000)
        & df["psf"].between(0.40, 15.0)
    ].copy()

    # 3. Structural numerics & missingness indicators
    df["rooms_num"] = df["rooms"].apply(parse_rooms)
    df["bathroom_num"] = df["bathroom"].apply(extract_number)
    parking_raw = df["parking"].apply(extract_number)
    df["has_parking_info"] = parking_raw.notna().astype(int)
    df["parking_num"] = parking_raw.fillna(0.0)

    df = df[
        df["rooms_num"].notna()
        & (df["rooms_num"] <= 6)
        & df["bathroom_num"].notna()
        & (df["bathroom_num"] <= 6)
        & (df["parking_num"] <= 4)
        & ~((df["size_sqft"] < 400) & (df["rooms_num"] >= 4))
    ].copy()

    df["completion_year_num"] = df["completion_year"].apply(extract_number)
    df["has_completion_year"] = df["completion_year_num"].notna().astype(int)

    # 4. Location (region + clean township area; map numeric Mudah area IDs to "other")
    df["region"] = df["region"].fillna("unknown").str.strip().str.lower()
    loc_split = df["location"].fillna("Unknown - Unknown").str.split(" - ", n=1, expand=True)
    df["area"] = loc_split[1].fillna(loc_split[0]).str.strip().str.lower()
    df["area"] = df["area"].where(~df["area"].str.isdigit(), "other")

    # 5. Categoricals
    df["property_type"] = df["property_type"].fillna("others").str.strip().str.lower()
    df["property_type"] = df["property_type"].where(
        df["property_type"].isin(KNOWN_PROPERTY_TYPES), "others"
    )
    df["furnished"] = (
        df["furnished"]
        .fillna("not furnished")
        .str.strip()
        .str.lower()
        .replace({"unknown": "not furnished"})
    )

    # 6. Facility & additional_facility features (excluding numeric scraper noise)
    fac_lower = df["facilities"].fillna("").str.lower()
    add_lower = df["additional_facilities"].fillna("").str.lower()

    df["facility_count"] = df["facilities"].apply(count_valid_facilities)
    df["has_pool"] = fac_lower.str.contains("swimming pool", regex=False).astype(int)
    df["has_gym"] = fac_lower.str.contains("gymnasium", regex=False).astype(int)
    df["has_security"] = fac_lower.str.contains("security", regex=False).astype(int)
    df["has_sauna"] = fac_lower.str.contains("sauna", regex=False).astype(int)
    df["has_lift"] = fac_lower.str.contains("lift", regex=False).astype(int)

    df["near_rail"] = add_lower.str.contains(
        "near ktm/lrt|railway|lrt|mrt|ktm|monorail", regex=True
    ).astype(int)
    df["has_aircond"] = add_lower.str.contains("air-cond", regex=False).astype(int)
    df["has_washer"] = add_lower.str.contains("washing machine", regex=False).astype(int)
    df["cooking_allowed"] = add_lower.str.contains("cooking allowed", regex=False).astype(int)
    df["has_internet"] = add_lower.str.contains("internet", regex=False).astype(int)

    return df[list(FEATURE_COLS) + ["monthly_rent"]].reset_index(drop=True)


def fit_preprocessor(train_df: pd.DataFrame, top_n_areas: int = TOP_N_AREAS) -> dict[str, Any]:
    """Fit top-N area vocabulary and completion year median strictly on training data."""
    top_areas = train_df["area"].value_counts().head(top_n_areas).index.tolist()
    year_median = float(train_df["completion_year_num"].median())
    return {
        "top_areas": set(top_areas),
        "top_areas_list": top_areas,
        "year_median": year_median,
    }


def transform_features(df: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted area bucketing and year median imputation to a DataFrame."""
    out = df[list(FEATURE_COLS)].copy()
    out["area"] = out["area"].where(out["area"].isin(prep["top_areas"]), "other")
    out["completion_year_num"] = out["completion_year_num"].fillna(prep["year_median"])
    return out


def eval_predictions(
    y_true_raw: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, float]:
    """Evaluate log-target predictions on raw MYR and log scales."""
    y_pred_raw = np.clip(np.expm1(y_pred_log), a_min=0.0, a_max=None)
    abs_err = np.abs(y_pred_raw - y_true_raw)
    return {
        "rmse_myr": round(float(root_mean_squared_error(y_true_raw, y_pred_raw)), 2),
        "mae_myr": round(float(mean_absolute_error(y_true_raw, y_pred_raw)), 2),
        "rmse_log": round(float(root_mean_squared_error(y_true_log, y_pred_log)), 4),
        "median_ae_myr": round(float(np.median(abs_err)), 2),
        "mdape_pct": round(float(np.median(abs_err / y_true_raw) * 100), 2),
    }


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict[str, Any]:
    """Run full training pipeline on 80% stratified split, evaluate holdout test, and export artifacts."""
    df_raw = pd.read_csv(data_path)
    df_clean = prepare_data(df_raw)
    print(f"Loaded {len(df_raw):,} raw rows -> {len(df_clean):,} cleaned listings ({len(FEATURE_COLS)} features)")

    rent_bins = pd.qcut(np.log1p(df_clean["monthly_rent"]), q=10, labels=False)
    df_full_train, df_test = train_test_split(
        df_clean, test_size=0.20, random_state=RANDOM_STATE, stratify=rent_bins
    )

    prep_artifacts = fit_preprocessor(df_full_train, top_n_areas=TOP_N_AREAS)
    dv = DictVectorizer(sparse=False)

    X_full_train = dv.fit_transform(
        transform_features(df_full_train, prep_artifacts).to_dict(orient="records")
    )
    X_test = dv.transform(
        transform_features(df_test, prep_artifacts).to_dict(orient="records")
    )

    y_full_train_raw = df_full_train["monthly_rent"].values
    y_full_train_log = np.log1p(y_full_train_raw)
    y_test_raw = df_test["monthly_rent"].values
    y_test_log = np.log1p(y_test_raw)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_full_train, y_full_train_log)

    train_metrics = eval_predictions(y_full_train_raw, y_full_train_log, model.predict(X_full_train))
    test_metrics = eval_predictions(y_test_raw, y_test_log, model.predict(X_test))

    print(
        f"Full Train (n={len(df_full_train):,}) -> "
        f"RMSE: RM {train_metrics['rmse_myr']:,.2f} | MAE: RM {train_metrics['mae_myr']:,.2f} | "
        f"Log-RMSE: {train_metrics['rmse_log']:.4f}"
    )
    print(
        f"Holdout Test (n={len(df_test):,}) -> "
        f"RMSE: RM {test_metrics['rmse_myr']:,.2f} | MAE: RM {test_metrics['mae_myr']:,.2f} | "
        f"Log-RMSE: {test_metrics['rmse_log']:.4f} | Median AE: RM {test_metrics['median_ae_myr']:,.2f} | "
        f"MdAPE: {test_metrics['mdape_pct']:.2f}%"
    )

    artifact = {
        "prep_artifacts": prep_artifacts,
        "dv": dv,
        "model": model,
        "feature_cols": list(FEATURE_COLS),
    }
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "model_type": "XGBRegressor",
        "target_transform": "log1p",
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
        "train_rows": int(len(df_full_train)),
        "test_rows": int(len(df_test)),
        "vectorized_features": int(X_full_train.shape[1]),
        "top_n_areas": TOP_N_AREAS,
        "top_areas_count": int(len(prep_artifacts["top_areas_list"])),
        "completion_year_median": prep_artifacts["year_median"],
        "xgb_params": XGB_PARAMS,
        "train_metrics": train_metrics,
        "holdout_test_metrics": test_metrics,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
        "model_sha256": model_sha256,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model bundle to {model_path.name} (sha256={model_sha256[:12]}...) and {metadata_path.name}")
    return metadata


if __name__ == "__main__":
    train_and_save()
