Claro, aquí está el texto traducido al español:

# Comparación de Métodos de Ensamble: Boosting vs. Bagging

Este proyecto realiza un análisis comparativo detallado de dos potentes técnicas de ensamble, Bagging (Random Forest) y Boosting (AdaBoost y Gradient Boosting), para resolver un problema de predicción de altos ingresos.

-----

### 🎯 **Objetivo Principal**

  - **Propósito:** Aplicar, evaluar y comparar el rendimiento de los modelos Random Forest, AdaBoost y Gradient Boosting en la misma tarea de clasificación.
  - **Problema de Negocio:** Determinar el modelo más efectivo e interpretable para predecir si una persona gana más de $50,000 anualmente basándose en datos demográficos.

### 📊 **Dataset Utilizado**

  - **Nombre:** Adult Income
  - **Fuente:** Cargado desde un archivo CSV local (`adult.csv`). Contiene datos demográficos y de empleo de la base de datos del Censo de 1994.

### 🛠️ **Metodología y Modelos Aplicados**

El proyecto sigue un flujo de trabajo estructurado de aprendizaje automático:

1.  **Análisis Exploratorio de Datos (EDA):** Análisis inicial y visualización de variables clave para comprender la estructura del dataset.
2.  **Preprocesamiento Robusto:** Uso de un `ColumnTransformer` para manejar eficientemente tanto las características categóricas (con `OneHotEncoder`) como las numéricas (con `StandardScaler`).
3.  **Entrenamiento de Modelos:** Se implementaron tres modelos de ensamble:
      - **Bagging:** `RandomForestClassifier`
      - **Boosting:**
          - `AdaBoostClassifier`
          - `GradientBoostingClassifier`
4.  **Evaluación de Rendimiento:** Los modelos se evaluaron en base a métricas clave de clasificación, incluyendo **Exactitud (Accuracy)**, **Reporte de Clasificación** (Precisión, Recall, F1-Score) y **Matriz de Confusión**.

### 🚀 **Resultados y Hallazgos Principales**

  - **Superioridad de los Métodos de Ensamble:** Los tres modelos de ensamble demostraron un fuerte rendimiento predictivo, superando significativamente a un modelo de referencia simple.
  - **Mejor Rendimiento 🏆:** **Gradient Boosting** resultó ser el modelo ganador, alcanzando la **exactitud (accuracy) más alta, de aproximadamente 87.4%**.
  - **Competidores Cercanos:** `Random Forest` y `AdaBoost` fueron altamente competitivos, con exactitudes de 86.1% y 86.4%, respectivamente, demostrando que las tres son técnicas viables.
  - **Variables Más Predictivas:** En los modelos exitosos, características clave como **edad**, **ganancias de capital**, **estado civil** y **horas trabajadas por semana** mostraron consistentemente la mayor importancia en la predicción de los niveles de ingresos.

### 🏆 **Recomendación Final**

Para esta tarea de predicción, se recomienda la implementación del modelo **Gradient Boosting**. Proporciona una ventaja de rendimiento ligera pero clara sobre los otros dos métodos, asegurando la clasificación más precisa de los rangos de ingresos.

# [**Ir al Proyecto**](../Actividad_2_Comparativa_Regresion/Actividad_3_Boosting_Bagging/Actividad_3_Modulo_5_Comparación_métodos_Boosting_y_Bagging_predicción_ingresos.ipynb)

-----

## ⚙️ **Cómo Ejecutar el Notebook**

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o una versión superior.**
2.  **Clona o descarga este repositorio.** Asegúrate de que el archivo del dataset `adult.csv` esté en el mismo directorio que el notebook.
3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Ejecuta el Jupyter Notebook:**
    `Actividad_3_Modulo_5_Comparación_métodos_Boosting_y_Bagging_predicción_ingresos.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos Avanzados](../README.md) | [Actividad Siguiente →](../Actividad_4_Validacion_Cruzada/README.md)
