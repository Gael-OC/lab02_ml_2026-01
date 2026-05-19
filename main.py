from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import LabConfig


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos del orquestador principal."""

    parser = argparse.ArgumentParser(
        description=(
            "Laboratorio 02: clasificacion de genero y regresion de edad con UTKFace."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help=(
            "Ruta a la carpeta que contiene las imagenes de UTKFace. "
            "Por defecto se usa ./dataset"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Carpeta donde se guardaran los modelos, reportes y figuras.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(25, 25),
        help="Tamano del rostro preprocesado. Ejemplo: --img-size 25 25",
    )
    parser.add_argument(
        "--no-oval-mask",
        action="store_false",
        dest="use_oval_mask",
        help="Desactiva la mascara oval durante el preprocesamiento.",
    )
    parser.add_argument(
        "--mask-scale-x",
        type=float,
        default=0.38,
        help="Escala horizontal de la mascara oval.",
    )
    parser.add_argument(
        "--mask-scale-y",
        type=float,
        default=0.48,
        help="Escala vertical de la mascara oval.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporcion del conjunto de prueba.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para mantener resultados reproducibles.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=[30, 50, 80, 100, 150, 200],
        help="Lista de componentes PCA a evaluar.",
    )
    parser.add_argument(
        "--gender-model",
        type=str,
        choices=["gaussian_nb", "lda"],
        default="gaussian_nb",
        help="Modelo para clasificacion de genero: gaussian_nb o lda.",
    )
    parser.add_argument(
        "--age-model",
        type=str,
        choices=["linear", "ridge"],
        default="linear",
        help="Modelo para regresion de edad: linear o ridge.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limite opcional para pruebas rapidas con menos imagenes.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LabConfig:
    """Construye el objeto de configuracion del laboratorio."""

    return LabConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.img_size),
        use_oval_mask=args.use_oval_mask,
        mask_scale_x=args.mask_scale_x,
        mask_scale_y=args.mask_scale_y,
        test_size=args.test_size,
        random_state=args.random_state,
        pca_components=tuple(args.pca_components),
        gender_model=args.gender_model,
        age_model=args.age_model,
        max_images=args.max_images,
    )


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    """Guarda un diccionario JSON de metricas en disco."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el flujo completo del clasificador de genero."""

    args = parse_args()

    from src.classification import (
        evaluate_gender_classifier,
        save_gender_classifier,
        split_dataset,
        train_gender_classifier,
    )
    from src.data import build_dataset, dataset_to_dataframe
    from src.regression import (
        evaluate_age_regressor,
        save_age_regressor,
        train_age_regressor,
    )
    from src.visualization import (
        save_confusion_matrix_figure,
        save_dataset_distribution_figure,
        save_pca_projection_figure,
    )

    config = build_config(args)
    config.ensure_output_dirs()

    print("[1/7] Cargando y preprocesando el dataset...")
    dataset = build_dataset(config)

    if len(dataset) == 0:
        raise RuntimeError(
            "No fue posible construir el dataset. "
            "Revise la ruta y el formato de nombres del dataset UTKFace."
        )

    print(
        f"    Muestras validas: {len(dataset)} | "
        f"Imagenes omitidas: {len(dataset.skipped_files)}"
    )

    dataset_df = dataset_to_dataframe(dataset)
    dataset_df.to_csv(config.reports_dir / "resumen_dataset.csv", index=False)
    save_dataset_distribution_figure(
        y_gender=dataset.y_gender,
        y_age=dataset.y_age,
        gender_map=config.gender_map,
        output_path=config.figures_dir / "distribucion_dataset.png",
    )

    print("[2/7] Separando entrenamiento y prueba...")
    split = split_dataset(
        dataset=dataset,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    print(
        "    Train:",
        split.X_train.shape,
        "| Test:",
        split.X_test.shape,
    )

    experiment_config = {
    "dataset_dir": str(config.dataset_dir),
    "output_dir": str(config.output_dir),
    "image_size": list(config.image_size),
    "use_oval_mask": config.use_oval_mask,
    "mask_scale_x": config.mask_scale_x,
    "mask_scale_y": config.mask_scale_y,
    "test_size": config.test_size,
    "random_state": config.random_state,
    "pca_components": list(config.pca_components),
    "gender_model": config.gender_model,
    "age_model": config.age_model,
    "max_images": config.max_images,
    "total_samples": len(dataset),
    "train_samples": int(split.X_train.shape[0]),
    "test_samples": int(split.X_test.shape[0]),
    }

    print("[3/7] Entrenando clasificador de genero ({config.gender_model}) con PCA...")
    training_result = train_gender_classifier(
        split=split,
        pca_components=config.pca_components,
        random_state=config.random_state,
        gender_model=config.gender_model,
    )
    print(f"    Componentes PCA probados: {training_result.pca_components_tested}")
    print(f"    Mejor configuracion: {training_result.grid_search.best_params_}")

    print("[4/7] Evaluando clasificador...")
    evaluation = evaluate_gender_classifier(
        model=training_result.best_estimator,
        split=split,
        best_params=training_result.grid_search.best_params_,
        best_cv_score=training_result.grid_search.best_score_,
    )

    print(
        "    Accuracy={:.4f} | Balanced Acc={:.4f} | Precision={:.4f} | "
        "Recall={:.4f} | F1={:.4f} | ROC-AUC={}".format(
            evaluation.accuracy,
            evaluation.balanced_accuracy,
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
            "N/A" if evaluation.roc_auc is None else f"{evaluation.roc_auc:.4f}",
        )
    )

    print("[5/7] Entrenando regresor de edad ({config.age_model}) con PCA...")
    age_training_result = train_age_regressor(
        X_train=split.X_train,
        y_age_train=split.y_age_train,
        pca_components=config.pca_components,
        random_state=config.random_state,
        age_model=config.age_model,
    )
    print(f"    Componentes PCA probados: {age_training_result.param_grid['pca__n_components']}")
    print(f"    Mejor configuracion: {age_training_result.best_params_}")

    print("[6/7] Evaluando regresor de edad...")
    age_evaluation = evaluate_age_regressor(
        model=age_training_result.best_estimator_,
        X_test=split.X_test,
        y_age_test=split.y_age_test,
    )

    age_metrics = {
    **age_evaluation,
    "best_params": age_training_result.best_params_,
    "best_cv_score": float(age_training_result.best_score_),
    "experiment_config": experiment_config,
    }

    print(
        "    MAE={:.4f} | MedAE={:.4f} | RMSE={:.4f} | R2={:.4f} | "
        "Predicciones negativas={}".format(
            age_metrics["mae"],
            age_metrics["median_absolute_error"],
            age_metrics["rmse"],
            age_metrics["r2"],
            age_metrics["negative_predictions"],
        )
    )

    print("[7/7] Guardando artefactos...")
    save_gender_classifier(
        model=training_result.best_estimator,
        output_path=config.models_dir / "pipeline_genero.pkl",
    )
    gender_metrics = evaluation.as_dict()
    gender_metrics["experiment_config"] = experiment_config

    save_metrics(
        metrics=gender_metrics,
        output_path=config.reports_dir / "metricas_genero.json",
    )
    save_age_regressor(
        model=age_training_result.best_estimator_,
        output_path=config.models_dir / "pipeline_edad.pkl",
    )
    save_metrics(
        metrics=age_metrics,
        output_path=config.reports_dir / "metricas_edad.json",
    )
    save_confusion_matrix_figure(
        confusion_matrix=evaluation.confusion_matrix,
        labels=[
            config.gender_map.get(0, "Clase 0"),
            config.gender_map.get(1, "Clase 1"),
        ],
        output_path=config.figures_dir / "matriz_confusion_genero.png",
    )
    save_pca_projection_figure(
        X=split.X_train,
        y=split.y_gender_train,
        gender_map=config.gender_map,
        random_state=config.random_state,
        output_path=config.figures_dir / "proyeccion_pca_genero.png",
    )

    print("Proceso completado.")
    print(f"Modelo de genero guardado en: {config.models_dir / 'pipeline_genero.pkl'}")
    print(f"Modelo de edad guardado en: {config.models_dir / 'pipeline_edad.pkl'}")
    print(f"Metricas de genero en: {config.reports_dir / 'metricas_genero.json'}")
    print(f"Metricas de edad en: {config.reports_dir / 'metricas_edad.json'}")
    print("Revise tambien src/streamlit_app.py para completar la app visual.")


if __name__ == "__main__":
    main()
