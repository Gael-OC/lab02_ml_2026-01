from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    max_error,
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class AgeRegressionGuide:
    """Resume los pasos que deben seguir los estudiantes para la regresion."""

    scoring: str = "neg_mean_absolute_error"
    suggested_metrics: tuple[str, ...] = ("MAE", "RMSE", "R2")

    def to_text(self) -> str:
        """Genera un recordatorio corto para acompanar el laboratorio."""

        return (
            "Guia de regresion de edad\n"
            "========================\n"
            "\n"
            "La parte de regresion no se implementa en esta version.\n"
            "Los estudiantes deben completar src/regression.py siguiendo estos pasos:\n"
            "\n"
            "1. Reutilizar la misma matriz X preprocesada.\n"
            "2. Construir un Pipeline con PCA + LinearRegression.\n"
            "3. Ajustar pca__n_components con GridSearchCV.\n"
            "4. Evaluar con MAE, RMSE y R2.\n"
            "5. Guardar el modelo cuando este listo.\n"
        )


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


def build_age_regression_pipeline(random_state: int) -> Any:
    # Construye el pipeline PCA + LinearRegression para predecir edad.
    return Pipeline(
        [
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("reg", LinearRegression()),
        ]
    )


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
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

    pipeline = build_age_regression_pipeline(random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={"pca__n_components": safe_components},
        scoring="neg_mean_absolute_error",
        cv=cv_folds,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_age_train)

    return grid_search


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, float]:
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_age_test, y_pred)
    medae = median_absolute_error(y_age_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_age_test, y_pred))
    r2 = r2_score(y_age_test, y_pred)
    worst_error = max_error(y_age_test, y_pred)

    negative_predictions = int(np.sum(y_pred < 0))
    negative_prediction_rate = negative_predictions / len(y_pred)

    return {
        "mae": float(mae),
        "median_absolute_error": float(medae),
        "rmse": float(rmse),
        "r2": float(r2),
        "max_error": float(worst_error),
        "negative_predictions": negative_predictions,
        "negative_prediction_rate": float(negative_prediction_rate),
    }


def save_age_regressor(model: Any, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)