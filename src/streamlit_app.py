from __future__ import annotations

from pathlib import Path

import streamlit as st


def run_app() -> None:
    """Ejecuta una app visual minima para la fase de deployment."""

    st.set_page_config(page_title="Lab02 ML - Demo visual", layout="centered")
    st.title("Laboratorio 02: demo visual minima")
    st.write(
        "Esta version solo carga una imagen y deja indicado donde deben integrarse "
        "el detector de caras y los modelos del laboratorio."
    )
    st.info(
        "Pendiente para estudiantes: agregar un detector de caras, recortar cada rostro "
        "y reutilizar el mismo preprocesamiento antes de inferir."
    )

    uploaded_file = st.file_uploader(
        "Sube una fotografia",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.stop()

    st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
    st.success("La imagen se cargo correctamente.")

    st.subheader("Siguiente trabajo para el laboratorio")
    st.markdown(
        "- Agregar un detector de caras.\n"
        "- Recortar cada rostro detectado.\n"
        "- Cargar el modelo de genero entrenado.\n"
        "- Implementar y cargar el modelo de edad.\n"
        "- Mostrar una prediccion por cada rostro."
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
            f"""# TODO(estudiantes): agregar detector de caras.
# Ejemplo posible con OpenCV:
# detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
#
# TODO(estudiantes): cargar aqui los modelos del laboratorio.
# gender_model = joblib.load("{suggested_gender_model}")
# age_model = joblib.load("{suggested_age_model}")
#
# TODO(estudiantes):
# 1. detectar rostros,
# 2. recortar cada rostro,
# 3. aplicar preprocess_face_array(...),
# 4. usar gender_model.predict(...),
# 5. usar age_model.predict(...),
# 6. mostrar la prediccion final en pantalla.
""",
            language="python",
        )

    st.warning(
        "La prediccion no se ejecuta todavia por diseno. "
        "Primero hay que incorporar el detector de caras y completar la regresion."
    )
