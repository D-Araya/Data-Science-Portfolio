# Análisis de Componentes Principales (PCA) para Reducción de Dimensionalidad

Este proyecto presenta un análisis detallado de la técnica de Reducción de Dimensionalidad mediante Análisis de Componentes Principales (PCA). Utilizando el dataset "Wine", el notebook demuestra un flujo de trabajo completo, desde la preparación y estandarización de los datos hasta la evaluación del impacto de PCA en un modelo predictivo.

-----

### 🎯 **Objetivo Principal**

  - **Propósito:** Demostrar la implementación, fundamentación teórica y aplicación práctica de PCA para simplificar un dataset multivariado, facilitar su visualización y evaluar su efecto en el rendimiento de un modelo de clasificación.
  - **Problema de Negocio:** Determinar si es posible reducir la complejidad de un dataset (disminuyendo el número de características) sin sacrificar, e incluso mejorando, la precisión de un modelo predictivo.

### 📊 **Dataset Utilizado**

  - **Nombre:** Wine
  - **Fuente:** Repositorio de datasets de `scikit-learn`.

### 🛠️ **Metodología y Modelos Aplicados**

El proyecto sigue un flujo de trabajo estructurado que incluye:

1.  **Análisis Exploratorio y Preprocesamiento:** Carga del dataset "Wine", seguido de una **estandarización** (`StandardScaler`) para asegurar que todas las características tengan la misma escala, un paso crucial para PCA.
2.  **Análisis de Varianza:** Se aplica PCA sobre el dataset completo para analizar la **varianza explicada** por cada componente principal y determinar el número óptimo de dimensiones a conservar.
3.  **Reducción y Visualización:** El dataset se reduce a **dos componentes principales** para visualizar la separabilidad de las clases de vino en un plano 2D mediante gráficos de dispersión, densidad y cajas.
4.  **Evaluación Comparativa:** Se compara el rendimiento (medido con **Accuracy**) de dos modelos de clasificación `K-Nearest Neighbors (KNN)`:
      - **Modelo Base:** Entrenado con las 13 características originales estandarizadas.
      - **Modelo con PCA:** Entrenado únicamente con los 2 componentes principales extraídos.

### 🚀 **Resultados y Hallazgos Principales**

  - **Efectividad de la Reducción:** Los dos primeros componentes principales (PC1 y PC2) lograron capturar el **55.4% de la varianza total**, demostrando ser suficientes para una excelente separación visual de las tres clases de vino.
  - **Poder de Separación:** El análisis visual y de cajas reveló que:
      - **PC1** actúa como un gran diferenciador entre la `class_0` y la `class_1`.
      - **PC2** aísla eficazmente a la `class_2` del resto.
  - **Modelo Ganador 🏆:** Sorprendentemente, el **modelo con PCA (2 dimensiones)** fue el de mejor rendimiento, alcanzando una **precisión del 96.30%**, superando ligeramente al modelo base con 13 dimensiones (94.44%).
  - **Optimización del Modelo:** PCA no solo simplificó el modelo en un **84.6%** (de 13 a 2 características), sino que también mejoró su rendimiento al actuar como un filtro de ruido, eliminando variaciones menos relevantes.

### 🏆 **Recomendación Final**

Se recomienda implementar **PCA como un paso de preprocesamiento** para este dataset. La técnica no solo es una herramienta poderosa para la visualización, sino que optimiza el modelo de clasificación, haciéndolo más simple, eficiente y preciso al enfocarse en las características más informativas.

# [**Ir al Proyecto**](../Actividad_3_PCA/Actividad_3_Modulo_6_Reducción_de_dimensionalidad_con_PCA.ipynb)

---

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**
2.  **Clona o descarga este repositorio.**
3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_3_Modulo_6_Reducción_de_dimensionalidad_con_PCA.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos No Supervisados](../README.md) | [Actividad Siguiente →](../Actividad_4_Segmentacion_Anomalias/README.md)
