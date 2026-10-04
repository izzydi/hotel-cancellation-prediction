"""Leakage-aware baseline models for hotel booking cancellation prediction.

The target is separated before modelling, common booking/reservation identifiers are
excluded automatically, optional dataset-specific leakage fields can be excluded from
the CLI, and all learned preprocessing remains inside cross-validation pipelines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 1821
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "hotel_reservations.csv"
OUTPUT_DIR = ROOT / "outputs"

COMMON_IDENTIFIER_NAMES = {
    "id",
    "bookingid",
    "bookingreference",
    "reservationid",
    "reservationreference",
}


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def infer_target(columns: list[str], requested: str | None) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested target '{requested}' is not present in the dataset.")
        return requested

    normalized = {normalize_name(c): c for c in columns}
    for candidate in (
        "bookingstatus",
        "reservationstatus",
        "cancellationstatus",
        "iscanceled",
        "iscancelled",
    ):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("Could not infer the target column. Pass it explicitly with --target.")


def find_identifier_columns(columns: list[str]) -> list[str]:
    """Return common booking/reservation identifier fields that should not be predictors."""
    return [c for c in columns if normalize_name(c) in COMMON_IDENTIFIER_NAMES]


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def main(
    data_path: Path,
    requested_target: str | None,
    requested_drop_columns: list[str],
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. See data/README.md.")

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()
    target = infer_target(df.columns.tolist(), requested_target)
    df = df.dropna(subset=[target])

    x = df.drop(columns=target)
    y = df[target].astype(str)
    if y.nunique() < 2:
        raise ValueError("The cancellation target must contain at least two classes.")

    unknown_drops = [c for c in requested_drop_columns if c not in x.columns]
    if unknown_drops:
        raise ValueError(
            "Requested --drop-column fields are not present after target removal: "
            + ", ".join(unknown_drops)
        )

    identifier_columns = find_identifier_columns(x.columns.tolist())
    dropped_columns = list(dict.fromkeys(identifier_columns + requested_drop_columns))
    if dropped_columns:
        x = x.drop(columns=dropped_columns)
    if x.shape[1] == 0:
        raise ValueError("No predictors remain after excluding identifiers/leakage fields.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in x_train.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    logistic = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logistic_search = GridSearchCV(
        logistic,
        {"model__C": [0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    forest = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    forest_search = GridSearchCV(
        forest,
        {
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 8],
        },
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    results: dict[str, dict[str, object]] = {}
    for name, search in (
        ("logistic_regression", logistic_search),
        ("random_forest", forest_search),
    ):
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)
        results[name] = {
            "best_cv_balanced_accuracy": float(search.best_score_),
            "best_params": search.best_params_,
            "held_out_test": evaluate(y_test, predictions),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "target": target,
        "rows": int(len(df)),
        "classes": sorted(y.unique().tolist()),
        "dropped_predictor_columns": dropped_columns,
        "evaluation": results,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help=(
            "Predictor column to exclude because it is an identifier or would not be "
            "available at prediction time. Repeat for multiple columns."
        ),
    )
    args = parser.parse_args()
    main(args.data, args.target, args.drop_column)
