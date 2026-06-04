# Laboratorio 02 - Machine Learning

**Grupo**: 1
**Estudiantes**: Gael Ortega y Matías Vidal

Este proyecto corresponde al Laboratorio 02 de Machine Learning. El objetivo principal es trabajar con imágenes faciales del dataset **UTKFace** para resolver dos tareas:

1. Clasificación de género.
2. Regresión de edad.

El laboratorio implementa un flujo basado en preprocesamiento de imágenes, reducción de dimensionalidad con PCA, entrenamiento con validación cruzada, evaluación con métricas y una demo visual con Streamlit.

---

## 1. Dataset utilizado

El dataset utilizado es **UTKFace**. En este dataset, las etiquetas vienen codificadas en el nombre del archivo con el siguiente formato:

```text
[age]_[gender]_[race]_[date&time].jpg
```

Donde:

```text
age    -> edad de la persona, entre 0 y 116
gender -> 0 = Hombre, 1 = Mujer
race   -> categoria racial entre 0 y 4
date   -> fecha/hora de recoleccion de la imagen
```

En este laboratorio se usan principalmente las etiquetas de edad y género.

---

## 2. Estructura general del proyecto

```text
.
├── main.py
├── main_visual.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── classification.py
│   ├── regression.py
│   ├── inference.py
│   ├── streamlit_app.py
│   └── visualization.py
├── dataset/
├── artifacts/
└── README.md
```

Descripción de los archivos principales:

| Archivo | Descripción |
|---|---|
| `main.py` | Ejecuta el flujo completo de entrenamiento, evaluación y guardado de artefactos. |
| `main_visual.py` | Ejecuta la app visual de Streamlit. |
| `src/config.py` | Define la configuración general del laboratorio. |
| `src/data.py` | Carga el dataset y extrae edad/género desde los nombres de archivo. |
| `src/preprocessing.py` | Aplica el preprocesamiento facial. |
| `src/classification.py` | Entrena y evalúa modelos de clasificación de género. |
| `src/regression.py` | Entrena y evalúa modelos de regresión de edad. |
| `src/inference.py` | Carga modelos entrenados y genera predicciones. |
| `src/streamlit_app.py` | Implementa la app visual. |
| `src/visualization.py` | Genera gráficos de distribución, PCA y matriz de confusión. |

---

## 3. Preprocesamiento aplicado

Cada imagen facial pasa por el siguiente flujo:

```text
imagen original
-> escala de grises
-> redimensionamiento
-> ecualizacion de histograma
-> mascara oval
-> normalizacion a [0, 1]
-> vectorizacion
```

La máscara oval se utiliza para reducir ruido externo al rostro, como fondo, cuello, pelo o bordes de la imagen.

La configuración base de la máscara es:

```text
use_oval_mask = True
mask_scale_x = 0.38
mask_scale_y = 0.48
```

Esta configuración corresponde a la máscara original usada como base en el laboratorio.

---

## 4. Modelos implementados

### 4.1 Clasificación de género

Se implementaron dos alternativas:

```text
PCA + GaussianNB
PCA + LDA
```

Inicialmente se usó `GaussianNB`, siguiendo el enfoque base del laboratorio. Luego se agregó `LinearDiscriminantAnalysis` como alternativa, ya que LDA usa las etiquetas para buscar una separación más discriminante entre clases.

### 4.2 Regresión de edad

Se implementaron dos alternativas:

```text
PCA + LinearRegression
PCA + Ridge
```

Inicialmente se usó `LinearRegression`. Luego se agregó `Ridge`, que mantiene el enfoque de regresión lineal, pero incorpora regularización para intentar mejorar la generalización y reducir errores extremos.

---

## 5. Instalación y preparación

El proyecto utiliza un entorno de Conda definido en `environment.yml`.

Crear el entorno:

```bash
conda env create -f environment.yml
```

Activar el entorno:

```bash
conda activate lab02-ml-2026-01
```

Si el entorno ya existe y se agregaron nuevas dependencias, actualizarlo con:

```bash
conda env update -f environment.yml --prune
```

El dataset debe ubicarse en la carpeta:

```text
dataset/
```

No se recomienda subir el dataset al repositorio, ya que es un archivo local y puede pesar demasiado.

---

## 6. Entrenamiento del modelo

Para ejecutar el entrenamiento con la configuración final recomendada:

```bash
python main.py \
  --img-size 50 50 \
  --pca-components 30 40 50 80 100 150 200 250 300 350 400 \
  --gender-model lda \
  --age-model ridge
```

Esto genera artefactos en:

```text
artifacts/
├── models/
│   ├── pipeline_genero.pkl
│   └── pipeline_edad.pkl
├── reports/
│   ├── metricas_genero.json
│   ├── metricas_edad.json
│   └── resumen_dataset.csv
└── figures/
    ├── matriz_confusion_genero.png
    ├── distribucion_dataset.png
    └── proyeccion_pca_genero.png
```

---

## 7. Ejecución de la app visual

Primero se debe entrenar el modelo para generar los archivos `.pkl`.

Luego se ejecuta:

```bash
streamlit run main_visual.py
```

La app permite:

1. Subir una imagen.
2. Detectar o procesar un rostro.
3. Aplicar el mismo preprocesamiento usado en entrenamiento.
4. Cargar los modelos entrenados.
5. Mostrar género y edad estimada.

---

## 8. Métricas utilizadas

### 8.1 Clasificación de género

Se calcularon las siguientes métricas:

```text
accuracy
balanced_accuracy
precision
recall
specificity
f1
roc_auc
confusion_matrix
classification_report
```

Estas métricas permiten analizar no solo el acierto global, sino también el rendimiento por clase.

### 8.2 Regresión de edad

Se calcularon las siguientes métricas:

```text
MAE
Median Absolute Error
RMSE
R2
Max Error
Predicciones negativas
Tasa de predicciones negativas
```

Además, la regresión de edad se evaluó por rangos:

```text
menores de 16 años
entre 16 y 60 años
mayores de 60 años
```

El rango principal definido fue:

```text
16 a 60 años
```

Este rango se usó porque permite analizar el rendimiento en edades adultas no extremas. También permite separar el análisis de niños y adultos mayores, donde el modelo presentó mayor dificultad.

---

## 9. Experimentos realizados

Se realizaron varias pruebas para mejorar el rendimiento del modelo sin alejarse del enfoque clásico del laboratorio.

### 9.1 Tamaño de imagen y PCA

Se probaron distintos tamaños de imagen:

```text
25x25
32x32
50x50
```

También se probaron distintos valores para `pca__n_components`.

| Experimento | Género F1 | Género ROC-AUC | Edad MAE global | Edad MAE 16-60 | PCA género | PCA edad |
|---|---:|---:|---:|---:|---:|---:|
| 25x25 default | 0.7733 | 0.8603 | 10.70 | 8.53 | 30 | 200 |
| 32x32 default | 0.7754 | 0.8664 | 10.61 | 8.48 | 30 | 200 |
| 50x50 default | 0.7775 | 0.8664 | 10.61 | 8.47 | 30 | 200 |
| 50x50 PCA hasta 300 | 0.7789 | 0.8674 | 10.50 | 8.43 | 40 | 300 |
| 50x50 PCA hasta 400 | 0.7789 | 0.8674 | 10.43 | 8.37 | 40 | 400 |

El mejor resultado de esta etapa se obtuvo con:

```text
image_size = 50x50
pca_components = 30, 40, 50, 80, 100, 150, 200, 250, 300, 350, 400
```

---

## 10. Evaluación de la máscara oval

Se probaron cuatro configuraciones de máscara:

| Experimento | use_oval_mask | mask_scale_x | mask_scale_y |
|---|---:|---:|---:|
| Máscara actual | True | 0.38 | 0.48 |
| Máscara chica | True | 0.32 | 0.42 |
| Máscara grande | True | 0.45 | 0.55 |
| Sin máscara | False | 0.38 | 0.48 |

### 10.1 Resultados en clasificación de género

| Experimento | Accuracy | Balanced Accuracy | F1 | ROC-AUC | Mejor PCA |
|---|---:|---:|---:|---:|---:|
| Máscara actual | 0.7908 | 0.7900 | 0.7789 | 0.8674 | 40 |
| Máscara chica | 0.7741 | 0.7735 | 0.7625 | 0.8495 | 30 |
| Máscara grande | 0.7895 | 0.7884 | 0.7757 | 0.8675 | 40 |
| Sin máscara | 0.7642 | 0.7624 | 0.7450 | 0.8382 | 40 |

La máscara actual obtuvo el mejor rendimiento general para género. El caso sin máscara empeoró de forma clara el resultado, lo que muestra que la máscara sí ayuda a reducir ruido externo.

### 10.2 Resultados en regresión de edad

| Experimento | MAE global | MAE 16-60 | MAE >60 | Predicciones negativas | Mejor PCA |
|---|---:|---:|---:|---:|---:|
| Máscara actual | 10.43 | 8.37 | 21.33 | 92 | 400 |
| Máscara chica | 10.50 | 8.44 | 21.66 | 87 | 400 |
| Máscara grande | 10.52 | 8.45 | 21.53 | 97 | 300 |
| Sin máscara | 10.44 | 8.42 | 20.82 | 101 | 350 |

La máscara actual también obtuvo el mejor equilibrio general para edad, especialmente en MAE global y MAE dentro del rango 16 a 60 años.

Por esta razón se mantuvo como configuración final:

```text
use_oval_mask = True
mask_scale_x = 0.38
mask_scale_y = 0.48
```

---

## 11. Comparación de modelos de género

Se comparó el modelo base `PCA + GaussianNB` contra `PCA + LDA`.

| Modelo | Accuracy | Balanced Accuracy | F1 | ROC-AUC | Mejor PCA |
|---|---:|---:|---:|---:|---:|
| PCA + GaussianNB | 0.7908 | 0.7900 | 0.7789 | 0.8674 | 40 |
| PCA + LDA | 0.8387 | 0.8390 | 0.8335 | 0.9124 | 350 |

LDA mejoró de forma importante el rendimiento de género. Por esta razón, se seleccionó `PCA + LDA` como modelo final para clasificación de género.

La matriz de confusión final para género fue:

| Real / Predicho | Hombre | Mujer |
|---|---:|---:|
| Hombre | 2062 | 416 |
| Mujer | 349 | 1915 |

---

## 12. Comparación de modelos de edad

Se comparó `PCA + LinearRegression` contra `PCA + Ridge`.

| Modelo | MAE global | MedAE global | RMSE global | R2 global | MAE 16-60 | Predicciones negativas |
|---|---:|---:|---:|---:|---:|---:|
| PCA + LinearRegression | 10.43 | 8.22 | 13.62 | 0.5381 | 8.37 | 92 |
| PCA + Ridge | 10.43 | 8.22 | 13.62 | 0.5381 | 8.35 | 92 |

Ridge produjo una mejora pequeña en el rango 16 a 60 años y mantuvo resultados similares en las métricas globales. Como es una versión regularizada de la regresión lineal y no empeoró los resultados, se seleccionó `PCA + Ridge` como modelo final para edad.

Los mejores hiperparámetros fueron:

```text
pca__n_components = 400
reg__alpha = 100.0
```

---

## 13. Configuración final seleccionada

La configuración final del laboratorio fue:

```text
image_size = 50x50
use_oval_mask = True
mask_scale_x = 0.38
mask_scale_y = 0.48
gender_model = lda
age_model = ridge
pca_components = 30, 40, 50, 80, 100, 150, 200, 250, 300, 350, 400
```

Comando final recomendado:

```bash
python main.py \
  --img-size 50 50 \
  --pca-components 30 40 50 80 100 150 200 250 300 350 400 \
  --gender-model lda \
  --age-model ridge
```

---

## 14. Resultados finales

### 14.1 Clasificación de género

| Métrica | Valor |
|---|---:|
| Accuracy | 0.8387 |
| Balanced accuracy | 0.8390 |
| Precision | 0.8215 |
| Recall | 0.8458 |
| Specificity | 0.8321 |
| F1 | 0.8335 |
| ROC-AUC | 0.9124 |
| Mejor PCA | 350 |

### 14.2 Regresión de edad

| Métrica | Valor |
|---|---:|
| MAE global | 10.43 |
| Median Absolute Error | 8.22 |
| RMSE global | 13.62 |
| R2 global | 0.5381 |
| Max Error | 67.74 |
| Predicciones negativas | 92 |
| Tasa de predicciones negativas | 0.0194 |
| Mejor PCA | 400 |
| Ridge alpha | 100.0 |

### 14.3 Regresión de edad por rango etario

| Rango | Muestras | MAE | MedAE | RMSE | R2 | Predicciones negativas |
|---|---:|---:|---:|---:|---:|---:|
| Menores de 16 | 771 | 12.65 | 11.59 | 15.25 | -11.25 | 89 |
| 16 a 60 | 3471 | 8.35 | 6.86 | 10.68 | 0.0848 | 3 |
| Mayores de 60 | 500 | 21.41 | 20.74 | 24.64 | -5.45 | 0 |

El modelo de edad funciona mejor en el rango de 16 a 60 años. En adultos mayores el error aumenta considerablemente.

---

## 15. Pruebas visuales con Streamlit

Se probaron imágenes del dataset UTKFace directamente en la app visual.

### 15.1 Casos adultos

| Imagen | Género real | Género predicho | Edad real | Edad predicha | Error |
|---|---|---|---:|---:|---:|
| `25_1_2_...` | Mujer | Mujer | 25 | 19 | 6 |
| `35_0_0_...` | Hombre | Hombre | 35 | 42 | 7 |
| `48_0_1_...` | Hombre | Hombre | 48 | 54 | 6 |

En estos casos, el género fue predicho correctamente y el error de edad estuvo dentro de un rango razonable.

### 15.2 Casos extremos

| Imagen | Género real | Género predicho | Edad real | Edad predicha | Comentario |
|---|---|---|---:|---:|---|
| `5_0_3_...` | Hombre | Hombre | 5 | 7 | Correcto |
| `5_1_4_...` | Mujer | Mujer | 5 | 4 | Correcto |
| `75_1_0_...` | Mujer | Hombre | 75 | 59 | Error en género y edad |
| `80_0_0_...` | Hombre | Mujer | 80 | 40 | Error en género y edad |

Estas pruebas confirman que el modelo puede funcionar bien en algunos casos infantiles, pero presenta errores importantes en adultos mayores, lo cual coincide con las métricas por rango etario.

---

## 16. Limitaciones

El proyecto tiene las siguientes limitaciones:

- El modelo fue entrenado con UTKFace, que contiene rostros recortados y alineados.
- En fotos externas con cuerpo completo, fondo cargado o mala iluminación, el rendimiento puede bajar.
- La edad es más difícil de estimar que el género.
- El modelo de edad tiene mayor error en adultos mayores.
- Ridge no elimina completamente las predicciones negativas.
- La app visual depende de la calidad del recorte facial.
- El enfoque usa modelos clásicos; no utiliza redes neuronales profundas.

---

## 17. Conclusión

El laboratorio logró implementar un flujo completo de Machine Learning para clasificación de género y regresión de edad usando imágenes faciales.

Se partió con una línea base basada en:

```text
PCA + GaussianNB
PCA + LinearRegression
```

Luego se realizaron mejoras controladas y alineadas con el contenido del curso:

```text
ajuste de tamaño de imagen
búsqueda de componentes PCA
comparación de máscaras ovales
PCA + LDA para género
PCA + Ridge para edad
métricas por rango etario
```

La mejor configuración final fue:

```text
PCA + LDA para género
PCA + Ridge para edad
imagen 50x50
máscara oval original
PCA hasta 400 componentes
```

El modelo final obtuvo un rendimiento razonable para un enfoque clásico, especialmente en clasificación de género. La regresión de edad funciona mejor en edades adultas entre 16 y 60 años, pero presenta mayor error en niños y adultos mayores.

La app visual en Streamlit permite probar el modelo entrenado de forma interactiva, mostrando género y edad estimada por rostro.