from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    max_error,
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def resolve_regression_pca_components(
    candidates: tuple[int, ...],
    X_train: Any,
    cv_folds: int,
) -> tuple[int, ...]:
    # Filtra componentes PCA validos para regresion segun el tamano del dataset.

    n_samples, n_features = X_train.shape
    smallest_train_fold_size = n_samples - int(np.ceil(n_samples / cv_folds))
    max_allowed = min(n_features, smallest_train_fold_size)

    if max_allowed < 1:
        raise ValueError("No hay suficientes muestras para ajustar PCA en regresion.")

    valid_candidates = tuple(
        component for component in candidates if 1 <= component <= max_allowed
    )

    if valid_candidates:
        return valid_candidates

    return (max_allowed,)


def build_age_regression_pipeline(
    random_state: int,
    age_model: str = "linear",
) -> Any:
    # Construye el pipeline PCA + regresor para predecir edad.

    if age_model == "linear":
        regressor = LinearRegression()
    elif age_model == "ridge":
        regressor = Ridge(random_state=random_state)
    else:
        raise ValueError(f"Modelo de edad no soportado: {age_model}")

    return Pipeline(
        [
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("reg", regressor),
        ]
    )


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
    age_model: str = "linear",
) -> Any:
    # Entrena el regresor de edad usando GridSearchCV.
    cv_folds = min(5, len(y_age_train))

    if cv_folds < 2:
        raise ValueError("Se requieren al menos dos muestras para entrenar regresion.")

    safe_components = resolve_regression_pca_components(
        candidates=pca_components,
        X_train=X_train,
        cv_folds=cv_folds,
    )

    pipeline = build_age_regression_pipeline(
        random_state=random_state,
        age_model=age_model,
    )

    param_grid: dict[str, Any] = {
        "pca__n_components": safe_components,
    }

    if age_model == "ridge":
        param_grid["reg__alpha"] = (0.1, 1.0, 10.0, 100.0)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_age_train)

    return grid_search


def evaluate_age_subset(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float | int | None]:
    # Calcula metricas de edad para un subconjunto especifico.

    if len(y_true) == 0:
        return {
            "samples": 0,
            "mae": None,
            "median_absolute_error": None,
            "rmse": None,
            "r2": None,
            "max_error": None,
            "negative_predictions": 0,
            "negative_prediction_rate": None,
        }

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    worst_error = max_error(y_true, y_pred)

    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = None

    negative_predictions = int(np.sum(y_pred < 0))
    negative_prediction_rate = negative_predictions / len(y_pred)

    return {
        "samples": int(len(y_true)),
        "mae": float(mae),
        "median_absolute_error": float(medae),
        "rmse": float(rmse),
        "r2": None if r2 is None else float(r2),
        "max_error": float(worst_error),
        "negative_predictions": negative_predictions,
        "negative_prediction_rate": float(negative_prediction_rate),
    }


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, Any]:
    # Calcula metricas globales y por rangos de edad.

    y_pred = model.predict(X_test)

    global_metrics = evaluate_age_subset(
        y_true=y_age_test,
        y_pred=y_pred,
    )

    child_mask = y_age_test < 16
    main_range_mask = (y_age_test >= 16) & (y_age_test <= 60)
    older_adult_mask = y_age_test > 60

    metrics_by_age_range = {
        "under_16": evaluate_age_subset(
            y_true=y_age_test[child_mask],
            y_pred=y_pred[child_mask],
        ),
        "from_16_to_60": evaluate_age_subset(
            y_true=y_age_test[main_range_mask],
            y_pred=y_pred[main_range_mask],
        ),
        "over_60": evaluate_age_subset(
            y_true=y_age_test[older_adult_mask],
            y_pred=y_pred[older_adult_mask],
        ),
    }

    return {
        **global_metrics,
        "main_age_range": {
            "min_age": 16,
            "max_age": 60,
            "description": "Rango principal definido para analizar el rendimiento en edades adultas no extremas.",
        },
        "metrics_by_age_range": metrics_by_age_range,
    }


def save_age_regressor(model: Any, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)