# Reducción de Dimensionalidad con PCA para Clasificación de Vinos

Este proyecto demuestra la aplicación y el poder del Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de un conjunto de datos. El objetivo es simplificar un dataset complejo, pasando de 13 características a solo 2, y luego evaluar cómo esta reducción afecta el rendimiento de un modelo de clasificación.

---

### 🎯 **Objetivo Principal**
- **Propósito:** Aplicar la técnica de PCA para reducir un conjunto de datos multivariado, reteniendo la máxima varianza posible y facilitando la visualización de las clases.
- **Problema de Negocio:** Demostrar que es posible entrenar un modelo de machine learning de alta precisión utilizando un número significativamente menor de características, lo que reduce la complejidad del modelo y los costos computacionales.

### 📊 **Dataset Utilizado**
- **Nombre:** Wine Dataset
- **Fuente:** Un conjunto de datos sobre las propiedades químicas de diferentes vinos de Italia, cargado directamente desde la librería `sklearn.datasets`.

### 🛠️ **Metodología y Modelos Aplicados**
El flujo de trabajo se centra en la transformación de datos y la evaluación de un modelo de clasificación:
1.  **Preprocesamiento:** Los datos fueron estandarizados utilizando `StandardScaler`. Este es un paso crucial antes de aplicar PCA para asegurar que todas las variables contribuyan de manera equitativa.
2.  **Reducción de Dimensionalidad:** Se aplicó el **Análisis de Componentes Principales (PCA)** para transformar las 13 características fisicoquímicas originales en solo **2 componentes principales**.
3.  **Modelo de Clasificación:** Se entrenó un modelo de `LogisticRegression` utilizando únicamente los 2 componentes generados por PCA como características de entrada.
4.  **Evaluación y Visualización:** El rendimiento del clasificador se midió con un **reporte de clasificación** (precisión, recall, F1-score). Finalmente, se visualizó el resultado en un gráfico de dispersión que muestra la separación de las clases en el nuevo espacio 2D y las fronteras de decisión aprendidas por el modelo.

### 🚀 **Resultados y Hallazgos Principales**
- **Reducción Exitosa:** PCA redujo de manera efectiva la dimensionalidad del dataset de **13 características a solo 2**.
- **Retención de Varianza:** A pesar de la drástica reducción, los dos componentes principales lograron capturar y explicar aproximadamente el **56% de la varianza** total del conjunto de datos original.
- **Rendimiento Excepcional del Modelo 🏆:** El modelo de `LogisticRegression`, entrenado con solo 2 componentes, alcanzó una impresionante **exactitud (accuracy) del 97%** en el conjunto de prueba. Esto demuestra que la información más relevante para la clasificación se conservó con éxito.
- **Clara Separabilidad Visual:** El gráfico de dispersión de los dos componentes principales mostró una **separación visual casi perfecta** entre las tres clases de vino, validando la efectividad de PCA para crear un nuevo espacio de características altamente informativo.

### 🏆 **Recomendación Final**
- El **Análisis de Componentes Principales (PCA)** es una técnica extremadamente poderosa y eficiente para la reducción de dimensionalidad. Permite simplificar modelos complejos y visualizar datos de alta dimensionalidad sin sacrificar significativamente el rendimiento predictivo.
- La estrategia de combinar **PCA para la extracción de características** con un clasificador simple como la **Regresión Logística** es un enfoque robusto y eficaz para abordar problemas de clasificación con muchas variables.

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
