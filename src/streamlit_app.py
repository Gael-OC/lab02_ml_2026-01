from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from src.config import LabConfig
from src.inference import (
    load_age_pipeline,
    load_gender_pipeline,
    predict_age_from_face,
    predict_gender_from_face,
)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    # Convierte el archivo subido por Streamlit a una imagen compatible con OpenCV.

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("No se pudo leer la imagen subida.")

    return image_bgr


def detect_faces(image_bgr: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    # Detecta rostros usando Haar Cascade de OpenCV.

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(40, 40),
    )

    faces = list(faces)
    faces.sort(key=lambda face: face[2] * face[3], reverse=True)

    annotated_image = image_bgr.copy()

    for x, y, w, h in faces:
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    return annotated_image, faces


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    # Convierte una imagen BGR de OpenCV a RGB para mostrarla en Streamlit.

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource
def load_trained_models(gender_model_path: str, age_model_path: str):
    # Carga los modelos entrenados una sola vez mientras la app esta abierta.

    gender_model = load_gender_pipeline(gender_model_path)
    age_model = load_age_pipeline(age_model_path)

    return gender_model, age_model


def run_app() -> None:
    """Ejecuta una app visual minima para la fase de deployment."""

    st.set_page_config(page_title="Lab02 ML - Demo visual", layout="centered")
    st.title("Laboratorio 02: demo visual")
    st.write(
        "Esta version permite cargar una imagen, detectar rostros y estimar "
        "genero y edad usando los modelos entrenados del laboratorio."
    )
    st.info(
        "Antes de usar esta app, ejecuta python main.py para generar los modelos "
        "pipeline_genero.pkl y pipeline_edad.pkl."
    )
    config = LabConfig()

    gender_model_path = config.models_dir / "pipeline_genero.pkl"
    age_model_path = config.models_dir / "pipeline_edad.pkl"

    missing_models = [
        str(model_path)
        for model_path in (gender_model_path, age_model_path)
        if not model_path.exists()
    ]

    if missing_models:
        st.error(
            "No se encontraron los modelos entrenados. "
            "Primero ejecuta python main.py desde la raiz del proyecto."
        )
        st.write("Archivos faltantes:")
        for model_path in missing_models:
            st.code(model_path)
        st.stop()

    gender_model, age_model = load_trained_models(
        str(gender_model_path),
        str(age_model_path),
    )

    uploaded_file = st.file_uploader(
        "Sube una fotografia",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.stop()

    try:
        image_bgr = read_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.image(
        bgr_to_rgb(image_bgr),
        caption="Imagen cargada",
        use_container_width=True,
    )

    annotated_image, faces = detect_faces(image_bgr)

    if not faces:
        height, width = image_bgr.shape[:2]
        faces = [(0, 0, width, height)]
        annotated_image = image_bgr.copy()

        cv2.rectangle(
            annotated_image,
            (0, 0),
            (width, height),
            (0, 255, 0),
            2,
        )

        st.warning(
            "No se detectaron rostros con Haar Cascade. "
            "Se usara la imagen completa como rostro de respaldo."
        )
    else:
        st.success(f"Se detectaron {len(faces)} rostro(s).")

    st.image(
        bgr_to_rgb(annotated_image),
        caption="Rostros detectados",
        use_container_width=True,
    )

    st.subheader("Predicciones por rostro")

    for index, (x, y, w, h) in enumerate(faces, start=1):
        face_crop = image_bgr[y : y + h, x : x + w]

        gender_prediction = predict_gender_from_face(
            face_array=face_crop,
            pipeline=gender_model,
            image_size=config.image_size,
            gender_map=config.gender_map,
        )

        age_prediction = predict_age_from_face(
            face_array=face_crop,
            pipeline=age_model,
            image_size=config.image_size,
        )

        st.markdown(f"### Rostro {index}")
        st.image(
            bgr_to_rgb(face_crop),
            caption=f"Rostro {index}",
            width=180,
        )

        st.write(f"Genero estimado: **{gender_prediction.label_name}**")

        display_age = max(0, age_prediction.rounded_age)

        st.write(f"Edad estimada: **{display_age} años**")
        st.caption(f"Valor continuo predicho por el modelo: {age_prediction.age_value:.2f}")

    st.success("Predicciones generadas correctamente.")