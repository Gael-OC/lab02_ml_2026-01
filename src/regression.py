from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def build_age_regression_pipeline(random_state: int) -> Any:
    """Interfaz guia para la regresion de edad.

    TODO(estudiantes):
    - Importar `PCA`, `LinearRegression` y `Pipeline`.
    - Construir un pipeline equivalente al de clasificacion:

        Pipeline([
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("reg", LinearRegression()),
        ])

    - Mantener la misma idea metodologica usada en el clasificador.
    """

    raise NotImplementedError(
        "Completar build_age_regression_pipeline en el laboratorio."
    )


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
) -> Any:
    """Interfaz guia para ajustar el regresor de edad.

    TODO(estudiantes):
    - Crear el pipeline con `build_age_regression_pipeline`.
    - Configurar `GridSearchCV` usando `pca__n_components`.
    - Sugerencia de scoring: `neg_mean_absolute_error`.
    - Retornar el mejor estimador encontrado.
    """

    raise NotImplementedError("Completar train_age_regressor en el laboratorio.")


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, float]:
    """Interfaz guia para calcular metricas de regresion.

    TODO(estudiantes):
    - Obtener las predicciones con `model.predict(X_test)`.
    - Calcular MAE.
    - Calcular RMSE.
    - Calcular R2.
    - Retornar un diccionario con esas metricas.
    """

    raise NotImplementedError("Completar evaluate_age_regressor en el laboratorio.")


def save_age_regressor(model: Any, output_path: str) -> None:
    """Interfaz guia para guardar el modelo de edad.

    TODO(estudiantes):
    - Importar `joblib`.
    - Guardar el pipeline completo, no solo el regresor final.
    - Usar un nombre sugerido como `pipeline_edad.pkl`.
    """

    raise NotImplementedError("Completar save_age_regressor en el laboratorio.")
