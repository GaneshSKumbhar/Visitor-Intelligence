from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "SpecialDay", "Month", "OperatingSystems",
    "Browser", "Region", "TrafficType", "VisitorType", "Weekend",
]
# PageValues is deliberately excluded from model input (leakage prevention).
# It must NOT appear in FEATURE_COLUMNS — the form never collects it.
COUNT_COLUMNS = ["Administrative", "Informational", "ProductRelated"]
DURATION_COLUMNS = [
    "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"
]
RATE_COLUMNS = ["BounceRates", "ExitRates"]
TECHNICAL_COLUMNS = ["OperatingSystems", "Browser", "Region", "TrafficType"]


class InputValidationError(ValueError):
    """A session is invalid or incompatible with the fitted model."""


def ensure_artifact_files(paths: list[Any]) -> None:
    """Raise a safe, named missing-artifact error before an app operation."""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path.name)


def validate_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    *,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """Validate and return model-ready, ordered feature columns."""
    if frame.empty:
        raise InputValidationError("Add at least one session before scoring.")
    if len(frame) > max_rows:
        raise InputValidationError(f"Upload at most {max_rows:,} sessions at a time.")
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise InputValidationError("Missing required fields: " + ", ".join(missing))

    clean = frame.loc[:, FEATURE_COLUMNS].copy()
    numeric = COUNT_COLUMNS + DURATION_COLUMNS + RATE_COLUMNS + [
        "SpecialDay", "Weekend", *TECHNICAL_COLUMNS
    ]
    for col in numeric:
        if col not in clean.columns:
            continue
        values = pd.to_numeric(clean[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise InputValidationError(f"{col} must contain valid numeric values.")
        clean[col] = values

    for col in COUNT_COLUMNS + DURATION_COLUMNS + RATE_COLUMNS + ["SpecialDay"]:
        low, high = limits[col]
        invalid = (clean[col] < low) | (clean[col] > high)
        if invalid.any():
            if col in COUNT_COLUMNS + DURATION_COLUMNS:
                raise InputValidationError(f"{col} must be between {low:g} and {high:g} (observed dataset range).")
            raise InputValidationError(f"{col} must be between {low:g} and {high:g} (observed dataset range).")
    for col in COUNT_COLUMNS:
        if (clean[col] % 1 != 0).any():
            raise InputValidationError(f"{col} must be a whole number.")
        clean[col] = clean[col].astype(int)

    if not clean["Weekend"].isin([0, 1]).all():
        raise InputValidationError("Weekend must be Yes or No.")
    clean["Weekend"] = clean["Weekend"].astype(int)

    month_options = metadata["observed_months"]
    visitor_options = metadata["visitor_types"]
    if not clean["Month"].isin(month_options).all():
        raise InputValidationError("Choose a month supported by the trained dataset.")
    if not clean["VisitorType"].isin(visitor_options).all():
        raise InputValidationError("Choose a supported visitor type.")
    for col in TECHNICAL_COLUMNS:
        valid_codes = metadata["technical_category_options"][col]
        if not clean[col].isin(valid_codes).all():
            raise InputValidationError(f"{col} must be one of the available category codes.")
        clean[col] = clean[col].astype(int)
    return clean


def score_sessions(model: Any, frame: pd.DataFrame, metadata: dict[str, Any], ratio: int) -> pd.DataFrame:
    """Use the fitted sklearn pipeline and saved score thresholds; do not retrain."""
    if str(ratio) not in metadata["cost_thresholds"]:
        raise InputValidationError("Select one of the available cost scenarios: R=3, 5, or 10.")
    scores = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    bands = metadata["score_bands"]
    labels = np.select(
        [scores >= bands["high_cutoff"], scores >= bands["medium_cutoff"]],
        ["High", "Medium"], default="Low"
    )
    threshold = float(metadata["cost_thresholds"][str(ratio)])
    references = {key: value["conversion_rate"] for key, value in bands["rates"].items()}
    return pd.DataFrame({
        "model_score": scores,
        "intent_band": labels,
        "historical_band_reference": [references[label] for label in labels],
        "intervention_flag": scores >= threshold,
        "threshold": threshold,
    }, index=frame.index)
