# Comparación de técnicas avanzadas para predicción de ingresos, un problema de Regresión o Clasificación?

Este proyecto explora y compara el rendimiento de modelos de regresión y clasificación para predecir si un individuo gana más de $50,000 al año. El notebook demuestra un flujo de trabajo robusto utilizando `Pipelines` y `ColumnTransformer` de scikit-learn.

---

### 🎯 **Objetivo Principal**
- **Propósito:** Aplicar y comparar cuatro modelos avanzados (dos de regresión adaptados y dos de clasificación nativos) sobre un mismo problema para evaluar su rendimiento, adecuación e interpretabilidad.
- **Problema de Negocio:** Determinar el modelo más preciso y fiable para predecir la categoría de ingresos de una persona basándose en sus características demográficas y laborales.

### 📊 **Dataset Utilizado**
- **Nombre:** Adult Income
- **Fuente:** Repositorio de OpenML, cargado a través de `scikit-learn`.

### 🛠️ **Metodología y Modelos Aplicados**
El proyecto sigue un flujo de trabajo estructurado que incluye:
1.  **Preprocesamiento Robusto:** Uso de `Pipelines` para imputar valores faltantes (mediana/moda) y escalar/codificar variables numéricas y categóricas.
2.  **Entrenamiento de Modelos:** Se implementaron y compararon cuatro enfoques:
    - **Regresión (Adaptada a Clasificación):**
        - `ElasticNet`
        - `QuantileRegressor`
    - **Clasificación (Nativos):**
        - `RandomForestClassifier`
        - `XGBoostClassifier`
3.  **Evaluación Comparativa:** El rendimiento se midió con métricas clave como **Accuracy**, **Matriz de Confusión** y, fundamentalmente, el **Área Bajo la Curva ROC (AUC)**.

### 🚀 **Resultados y Hallazgos Principales**
- **Fracaso de los Modelos de Regresión:** Los modelos `Elastic Net` y `Regresión Cuantílica`, al ser adaptados para clasificación con un umbral de 0.5, **fallaron completamente** (ROC AUC de 0.50), demostrando no tener ninguna capacidad predictiva para este problema.
- **Superioridad de los Modelos de Árboles:** `Random Forest` y `XGBoost` demostraron ser altamente efectivos, logrando excelentes resultados.
- **Modelo Ganador 🏆:** **XGBoost** fue el modelo con mejor rendimiento, alcanzando la mayor precisión y un **ROC AUC de 0.929**.
- **Variables Más Predictivas:** Los modelos exitosos identificaron consistentemente el **estado civil**, las **ganancias de capital** y el **nivel educativo** como los factores de mayor impacto en la predicción de ingresos.

### 🏆 **Recomendación Final**
Se recomienda implementar el **modelo XGBoost** para la tarea de predicción de ingresos, debido a su superioridad demostrada en todas las métricas de clasificación y su capacidad para generar insights de negocio claros y accionables.

# [**Ir al Proyecto**](../Actividad_1_Comparacion_Tecnicas_Prediccion/Actividad_1_Modulo_5_Comp_técnicas_avanzadas_predicción_ingresos.ipynb)

---

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**

2.  **Clona o descarga este repositorio.**

3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```

4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_1_Modulo_5_Comp_técnicas_avanzadas_predicción_ingresos.ipynb`

---
[Volver al índice principal](../../README.md) | [Volver a Modelos Avanzados](../README.md) | [Actividad Siguiente →](../Actividad_2_Comparativa_Regresion/README.md)

