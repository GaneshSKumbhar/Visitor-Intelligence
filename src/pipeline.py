from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
SKEWED = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues",
]
PASSTHROUGH = ["SpecialDay", "Weekend"]
CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def make_preprocessor(drop: tuple[str, ...] = ()) -> ColumnTransformer:
    skewed = [c for c in SKEWED if c not in drop]
    cats = [c for c in CATEGORICAL if c not in drop]
    return ColumnTransformer([
        ("skewed", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), skewed),
        ("pass", "passthrough", PASSTHROUGH),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), cats),
    ])


def make_xgb(pos_weight: float, *, n_estimators: int = 265) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.037,
        max_depth=4,
        min_child_weight=9,
        subsample=0.741,
        colsample_bytree=0.953,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
    )


def make_model(pos_weight: float, *, drop_page_values: bool = True) -> Pipeline:
    # PageValues is deliberately excluded from the primary scoring model.
    drops = ("PageValues",) if drop_page_values else ()
    return Pipeline([
        ("prep", make_preprocessor(drop=drops)),
        ("clf", make_xgb(pos_weight)),
    ])
