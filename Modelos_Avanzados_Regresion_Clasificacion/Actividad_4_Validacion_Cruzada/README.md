# Aplicación y Comparación de Técnicas de Validación Cruzada

Este proyecto ofrece un análisis práctico sobre cómo diferentes estrategias de validación cruzada pueden afectar la evaluación de un modelo de clasificación. Se demuestra la importancia de elegir el método de validación adecuado para obtener una estimación robusta y fiable del rendimiento del modelo.

---

### 🎯 **Objetivo Principal**
- **Propósito:** Aplicar y comparar tres métodos de validación cruzada (`K-Fold`, `Stratified K-Fold` y `Leave-One-Out`) para evaluar la estabilidad y el rendimiento de un modelo de Regresión Logística.
- **Problema de Negocio:** Determinar cuál estrategia de validación cruzada es la más adecuada para un problema de clasificación con clases desbalanceadas, asegurando que la métrica de rendimiento (accuracy) sea confiable.

### 📊 **Dataset Utilizado**
- **Nombre:** `winequality-red.csv`
- **Fuente:** Un conjunto de datos sobre las propiedades fisicoquímicas de vinos tintos, cargado desde un archivo CSV local.

### 🛠️ **Metodología y Modelos Aplicados**
El flujo de trabajo se centra en la evaluación robusta de un único modelo de clasificación:
1.  **Preprocesamiento:** Se escalaron las características numéricas con `StandardScaler` y se transformó la variable objetivo (calidad del vino) en una clase binaria ("bueno" vs. "malo").
2.  **Modelo Base:** Se utilizó un modelo de `LogisticRegression` como clasificador a evaluar.
3.  **Estrategias de Validación Comparadas:** Se implementó y midió el rendimiento del modelo utilizando tres técnicas de validación cruzada:
    - `K-Fold Cross-Validation`
    - `Stratified K-Fold Cross-Validation` (diseñado para problemas de clasificación)
    - `Leave-One-Out Cross-Validation` (LOOCV)
4.  **Evaluación Comparativa:** El rendimiento de cada estrategia se comparó en función de la **exactitud media (mean accuracy)** y, fundamentalmente, la **desviación estándar** de los resultados, que indica la estabilidad de la estimación.

### 🚀 **Resultados y Hallazgos Principales**
- **Rendimiento Similar:** Todos los métodos de validación cruzada arrojaron una exactitud media muy similar, en torno al **74-75%**.
- **Inestabilidad de K-Fold:** La validación `K-Fold` estándar mostró la **mayor desviación estándar**, lo que indica que sus estimaciones de rendimiento son menos estables. Esto se debe a que no garantiza una distribución equitativa de las clases en cada pliegue.
- **Robustez de Stratified K-Fold 🏆:** El método `Stratified K-Fold` demostró ser el más robusto, ofreciendo una exactitud media alta con una **desviación estándar significativamente menor**. Al preservar la proporción de clases en cada pliegue, proporciona una evaluación mucho más fiable del rendimiento del modelo.
- **LOOCV como Benchmark:** `Leave-One-Out` ofrece la estimación teóricamente menos sesgada, pero su altísimo coste computacional lo hace inviable para datasets de tamaño considerable.

### 🏆 **Recomendación Final**
Para problemas de clasificación, especialmente con datasets que puedan tener clases desbalanceadas, se recomienda **implementar `Stratified K-Fold` como estrategia de validación estándar**. Este método ofrece el mejor equilibrio entre una estimación de rendimiento precisa, estabilidad en los resultados y eficiencia computacional.

# [**Ir al Proyecto**](../Actividad_4_Validacion_Cruzada/Actividad_4_Modulo_5_Aplicación_Técnicas_Validación_Cruzada.ipynb)

---

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**
2.  **Clona o descarga este repositorio.** El archivo `winequality-red.csv` debe estar en el mismo directorio que el notebook.
3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_4_Modulo_5_Aplicación_Técnicas_Validación_Cruzada.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos Avanzados](../README.md) | [Actividad Siguiente →](../Actividad_5_Regularizacion/README.md)
