# Análisis de Segmentación y Detección de Anomalías en Pacientes Crónicos

Este proyecto aplica un pipeline de aprendizaje no supervisado para segmentar una cohorte de pacientes con diabetes y detectar perfiles anómalos. El notebook demuestra un flujo de trabajo completo, desde el preprocesamiento de datos hasta la interpretación clínica de los hallazgos.

-----

### 🎯 **Objetivo Principal**

  - **Propósito:** Utilizar técnicas de reducción de dimensionalidad, clustering y detección de anomalías para traducir datos clínicos brutos en insights accionables que permitan una medicina más proactiva.
  - **Problema de Negocio:** Identificar grupos de pacientes con características similares (segmentación) y detectar individuos con perfiles atípicos (anomalías) para centrar los esfuerzos de los equipos de salud en casos específicos o diseñar intervenciones dirigidas.

### 📊 **Dataset Utilizado**

  - **Nombre:** PIMA Indians Diabetes
  - **Fuente:** Repositorio público en GitHub, cargado a través de la URL: `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`.

### 🛠️ **Metodología y Modelos Aplicados**

El proyecto sigue un flujo de trabajo estructurado que incluye:

1.  **Preprocesamiento Robusto:** Estandarización de las variables clínicas con `StandardScaler` para asegurar su comparabilidad.
2.  **Reducción de Dimensionalidad Comparativa:** Se aplicaron y compararon tres técnicas para la visualización en 2D:
      - `PCA`
      - `t-SNE`
      - `UMAP` (Seleccionado como el mejor método por su balance entre estructura local y global).
3.  **Segmentación por Densidad:** Se utilizaron algoritmos de clustering para agrupar a los pacientes:
      - `DBSCAN`
      - `HDBSCAN` (Seleccionado por su capacidad para identificar clústeres de forma robusta y manejar el ruido).
4.  **Detección de Anomalías:** Se emplearon dos modelos para identificar formalmente los perfiles atípicos:
      - `Isolation Forest`
      - `One-Class SVM`
5.  **Análisis Cruzado:** Se integraron los resultados del clustering y la detección de anomalías para validar los hallazgos.

### 🚀 **Resultados y Hallazgos Principales**

  - **Mejor Técnica de Visualización:** **UMAP** demostró ser la técnica más efectiva para representar la estructura de los datos, mostrando un buen equilibrio entre la separación de grupos y la conservación de la topología global.
  - **Segmentación Exitosa:** **HDBSCAN** identificó **dos clústeres principales** y un conjunto significativo de puntos considerados como "ruido" o atípicos, demostrando su superioridad sobre DBSCAN para este dataset.
  - **Modelo Ganador 🏆:** La combinación de **UMAP para visualización, HDBSCAN para clustering y Isolation Forest para detección de anomalías** fue el pipeline más eficaz, robusto e interpretable.
  - **Validación de Hallazgos:** Las anomalías identificadas por **Isolation Forest** coincidieron en gran medida con los puntos de ruido detectados por **HDBSCAN**, lo que valida la coherencia de ambos métodos.

### 🏆 **Recomendación Final**

Se recomienda implementar el pipeline compuesto por **UMAP, HDBSCAN e Isolation Forest** para tareas de análisis no supervisado en datos clínicos similares. Las acciones de negocio derivadas son:

  - **Diseñar intervenciones dirigidas** para los perfiles de pacientes "típicos" encontrados en los clústeres.
  - **Investigar los casos atípicos** identificados para detectar posibles errores en los datos o fenotipos de enfermedades raras que requieran un estudio más profundo.

# [**Ir al Proyecto**](../Actividad_4_Segmentacion_Anomalias/Actividad_4_Modulo_6_Análisis_Segmentación_y_Detección_Anomalías_Pacientes_Crónicos.ipynb)

-----

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**
2.  **Clona o descarga este repositorio.**
3.  **Instala las librerías necesarias:**
    ```bash
    pip install pandas numpy scikit-learn umap-learn hdbscan matplotlib seaborn
    ```
4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_4_Modulo_6_Análisis_Segmentación_y_Detección_Anomalías_Pacientes_Crónicos.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos No Supervisados](../README.md) | [Actividad Siguiente →](../../Modelos_Deep_Learning_Modernos/Actividad_1_Fashion_MNIST/README.md)
