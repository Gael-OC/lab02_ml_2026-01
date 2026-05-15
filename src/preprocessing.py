from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_oval_mask(
    shape: tuple[int, int],
    scale_x: float = 0.38,
    scale_y: float = 0.48,
) -> np.ndarray:
    """Construye una mascara oval simple para atenuar el fondo del rostro."""

    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(1, int(width * scale_x)), max(1, int(height * scale_y)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def preprocess_face_array(
    face_bgr_or_gray: np.ndarray,
    size: tuple[int, int] = (25, 25),
    use_oval_mask: bool = True,
    mask_scale_x: float = 0.38,
    mask_scale_y: float = 0.48,
) -> tuple[np.ndarray, np.ndarray]:
    """Transforma un rostro en un vector numerico listo para el modelo.

    Pasos:
    1. convertir a grises si hace falta,
    2. redimensionar,
    3. ecualizar histograma,
    4. aplicar mascara oval,
    5. normalizar en [0, 1],
    6. aplanar.
    """

    if face_bgr_or_gray is None:
        raise ValueError("La imagen recibida es None.")

    if face_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    elif face_bgr_or_gray.ndim == 2:
        gray = face_bgr_or_gray.copy()
    else:
        raise ValueError("La imagen debe ser 2D o 3D.")

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)

    if use_oval_mask:
        mask = build_oval_mask(
            gray.shape,
            scale_x=mask_scale_x,
            scale_y=mask_scale_y,
        )
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    gray_norm = gray.astype(np.float32) / 255.0

    return gray_norm.flatten(), gray_norm


def preprocess_image_path(
    image_path: str | Path,
    size: tuple[int, int] = (25, 25),
    use_oval_mask: bool = True,
    mask_scale_x: float = 0.38,
    mask_scale_y: float = 0.48,
) -> tuple[np.ndarray, np.ndarray]:
    """Lee una imagen desde disco y aplica el mismo preprocesamiento facial."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    return preprocess_face_array(
        image,
        size=size,
        use_oval_mask=use_oval_mask,
        mask_scale_x=mask_scale_x,
        mask_scale_y=mask_scale_y,
    )
