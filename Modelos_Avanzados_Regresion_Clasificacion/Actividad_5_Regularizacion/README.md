# Aplicación de Regularización en Modelos de Regresión Lineal

Este proyecto demuestra de manera práctica el efecto y los beneficios de aplicar técnicas de regularización (Ridge, Lasso y ElasticNet) para mejorar el rendimiento y la estabilidad de un modelo de regresión lineal, comparándolos contra un modelo base sin regularizar.

---

### 🎯 **Objetivo Principal**
- **Propósito:** Evaluar cómo la regularización ayuda a combatir el sobreajuste (overfitting) y la inestabilidad de los coeficientes en modelos lineales, especialmente en presencia de multicolinealidad.
- **Problema de Negocio:** Comparar el rendimiento predictivo (R² y RMSE) y la estructura de los coeficientes de los modelos `Ridge` (L2), `Lasso` (L1) y `ElasticNet` frente a una `Regresión Lineal` estándar en una tarea de predicción de precios de vivienda.

### 📊 **Dataset Utilizado**
- **Nombre:** California Housing
- **Fuente:** Cargado directamente desde la librería `sklearn.datasets`.

### 🛠️ **Metodología y Modelos Aplicados**
El análisis sigue un flujo de trabajo claro y metódico:
1.  **Preprocesamiento:** Los datos fueron divididos en conjuntos de entrenamiento y prueba. Las características numéricas se estandarizaron utilizando `StandardScaler` para asegurar que la regularización se aplicara de manera uniforme.
2.  **Modelos Comparados:** Se entrenaron y evaluaron cuatro modelos de regresión:
    - `LinearRegression` (como modelo de referencia sin regularización).
    - `Ridge` (Regularización L2), que penaliza la suma de los cuadrados de los coeficientes.
    - `Lasso` (Regularización L1), que penaliza la suma de los valores absolutos de los coeficientes.
    - `ElasticNet` (Combinación de L1 y L2).
3.  **Evaluación:** El rendimiento se midió con dos métricas clave: el **Coeficiente de Determinación (R²)** y la **Raíz del Error Cuadrático Medio (RMSE)**. Adicionalmente, se realizó un análisis visual comparativo de los coeficientes de cada modelo.

### 🚀 **Resultados y Hallazgos Principales**
- **Mejora del Rendimiento:** Todos los modelos con regularización (`Ridge`, `Lasso`, `ElasticNet`) **superaron al modelo de `Regresión Lineal` base**, obteniendo un R² ligeramente más alto y un RMSE más bajo, lo que demuestra una mejor capacidad de generalización.
- **Efecto de la Regularización en los Coeficientes:**
    - **Ridge 🏆:** Fue el modelo con el **mejor rendimiento predictivo**. Logró estabilizar el modelo al "encoger" los coeficientes, reduciendo su magnitud en comparación con la regresión lineal simple.
    - **Lasso:** Demostró su capacidad para la **selección de características**, llevando a cero los coeficientes de las variables menos importantes y simplificando el modelo final.
    - **Regresión Lineal (Base):** Mostró coeficientes extremadamente grandes e inestables, un claro indicador de sobreajuste y sensibilidad a la correlación entre variables.

### 🏆 **Recomendación Final**
- La regularización es una técnica fundamental y efectiva para mejorar la robustez de los modelos lineales.
- Se recomienda **`Ridge`** cuando el objetivo principal es la **máxima precisión predictiva** y se desea mantener todas las variables en el modelo.
- Se recomienda **`Lasso`** cuando el objetivo es obtener un **modelo más simple e interpretable**, ya que realiza una selección automática de las características más relevantes.


# [**Ir al Proyecto**](../Actividad_2_Comparativa_Regresion/Actividad_5_Regularizacion/Actividad_5_Modulo_5_Aplicación_Regularización_Modelo_Regresión.ipynb)


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
    `Actividad_5_Modulo_5_Aplicación_Regularización_Modelo_Regresión.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos Avanzados](../README.md) | [Actividad Siguiente →](../Actividad_6_Seleccion_Caracteristicas/README.md)
