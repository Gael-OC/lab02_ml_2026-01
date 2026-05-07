from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st


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

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    annotated_image = image_bgr.copy()

    for x, y, w, h in faces:
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    return annotated_image, list(faces)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    # Convierte una imagen BGR de OpenCV a RGB para mostrarla en Streamlit.

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def run_app() -> None:
    """Ejecuta una app visual minima para la fase de deployment."""

    st.set_page_config(page_title="Lab02 ML - Demo visual", layout="centered")
    st.title("Laboratorio 02: demo visual minima")
    st.write(
        "Esta version permite cargar una imagen, detectar rostros y mostrar "
        "los recortes encontrados antes de integrar los modelos del laboratorio."
    )
    st.info(
        "Pendiente para estudiantes: cargar los modelos entrenados de genero y edad, "
        "aplicar el mismo preprocesamiento y mostrar una prediccion por rostro."
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
        st.warning("No se detectaron rostros en la imagen.")
        st.stop()

    st.success(f"Se detectaron {len(faces)} rostro(s).")

    st.image(
        bgr_to_rgb(annotated_image),
        caption="Rostros detectados",
        use_container_width=True,
    )

    st.subheader("Rostros recortados")

    for index, (x, y, w, h) in enumerate(faces, start=1):
        face_crop = image_bgr[y : y + h, x : x + w]

        st.image(
            bgr_to_rgb(face_crop),
            caption=f"Rostro {index}",
            width=180,
        )

    suggested_gender_model = Path("artifacts/models/pipeline_genero.pkl")
    suggested_age_model = Path("artifacts/models/pipeline_edad.pkl")

    # TODO(estudiantes): aqui deben cargarse los modelos cuando la parte visual
    # del laboratorio incorpore detector de caras e inferencia real.
    # gender_model = joblib.load(suggested_gender_model)
    # age_model = joblib.load(suggested_age_model)
    #
    # TODO(estudiantes): tambien debe agregarse un detector de caras para obtener
    # cada rostro antes de llamar a preprocess_face_array(...).

    with st.expander("Guia de integracion para estudiantes"):
        st.code(
    f"""# Detector de caras ya integrado con OpenCV.
# El siguiente paso es cargar aqui los modelos del laboratorio.
# gender_model = joblib.load("{suggested_gender_model}")
# age_model = joblib.load("{suggested_age_model}")
#
# TODO(estudiantes):
# 1. cargar los modelos entrenados,
# 2. usar cada rostro recortado,
# 3. aplicar preprocess_face_array(...),
# 4. usar gender_model.predict(...),
# 5. usar age_model.predict(...),
# 6. mostrar genero y edad estimada por cada rostro.
""",
    language="python",
)

    st.warning(
        "La prediccion no se ejecuta todavia. "
        "El siguiente paso es cargar los modelos de genero y edad."
    )
