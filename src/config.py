from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LabConfig:
    """Configura los parametros generales del laboratorio."""

    dataset_dir: Path = Path("dataset")
    output_dir: Path = Path("artifacts")
    image_size: tuple[int, int] = (25, 25)
    use_oval_mask: bool = True
    mask_scale_x: float = 0.38
    mask_scale_y: float = 0.48
    test_size: float = 0.20
    random_state: int = 42
    pca_components: tuple[int, ...] = (30, 50, 80, 100, 150, 200)
    gender_model: str = "gaussian_nb"
    age_model: str = "linear"
    gender_map: dict[int, str] = field(
        default_factory=lambda: {0: "Mujer", 1: "Hombre"}
    )
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_images: int | None = None

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
        self.pca_components = tuple(sorted({int(value) for value in self.pca_components}))

        if not 0 < self.test_size < 1:
            raise ValueError("test_size debe quedar entre 0 y 1.")

        if self.max_images is not None and self.max_images <= 0:
            raise ValueError("max_images debe ser mayor que cero.")
        
        if not 0 < self.mask_scale_x <= 1:
            raise ValueError("mask_scale_x debe quedar entre 0 y 1.")

        if not 0 < self.mask_scale_y <= 1:
            raise ValueError("mask_scale_y debe quedar entre 0 y 1.")

        allowed_gender_models = {"gaussian_nb", "lda"}
        if self.gender_model not in allowed_gender_models:
            raise ValueError(
                f"gender_model debe ser uno de estos valores: {sorted(allowed_gender_models)}"
            )
        allowed_age_models = {"linear", "ridge"}
        if self.age_model not in allowed_age_models:
            raise ValueError(
                f"age_model debe ser uno de estos valores: {sorted(allowed_age_models)}"
            )

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def ensure_output_dirs(self) -> None:
        """Crea las carpetas necesarias para guardar artefactos."""

        for directory in (self.output_dir, self.models_dir, self.reports_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)
