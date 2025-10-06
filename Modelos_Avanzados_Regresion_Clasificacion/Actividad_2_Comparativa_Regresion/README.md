# Aplicación Comparativa de Técnicas Avanzadas de Regresión

Este proyecto presenta un análisis comparativo de tres técnicas avanzadas de regresión, cada una aplicada a un problema específico para el cual es particularmente adecuada: predicción de valor central (Elastic Net), estimación de rangos (Regresión Cuantílica) y pronóstico de series de tiempo multivariadas (VAR).

---

### 🎯 **Objetivo Principal**
- **Propósito:** Demostrar la aplicación práctica y las fortalezas distintivas de diferentes modelos de regresión en diversos casos de uso del mundo real.
- **Problema de Negocio:** Ilustrar cómo seleccionar la herramienta de regresión correcta según el objetivo analítico:
    1. Para predecir un **valor puntual** (precio de una vivienda).
    2. Para estimar un **rango de resultados probables** (horas trabajadas).
    3. Para pronosticar la **tendencia de un sistema** de variables interconectadas (indicadores macroeconómicos).

### 📊 **Datasets Utilizados**
- **California Housing:** Para la predicción de precios de viviendas (cargado desde `sklearn`).
- **Adult Income:** Para la estimación de horas trabajadas (cargado desde `OpenML`).
- **Macrodata:** Para el pronóstico de indicadores económicos (cargado desde `statsmodels`).

### 🛠️ **Metodología y Modelos Aplicados**
Se implementaron y evaluaron tres enfoques de regresión, cada uno con una metodología rigurosa:
1.  **Elastic Net (Precios de Viviendas):** Se utilizó `ElasticNetCV` para encontrar los hiperparámetros óptimos y predecir el valor mediano de las viviendas. El análisis se centró en la precisión (R², RMSE) y la interpretabilidad de los coeficientes.
2.  **Regresión Cuantílica (Horas Trabajadas):** Se entrenaron tres modelos para estimar los percentiles 10, 50 y 90 de las horas trabajadas, creando así un intervalo de predicción para cuantificar la incertidumbre.
3.  **Vector Autoregressive - VAR (Indicadores Económicos):** Se aplicó un enfoque econométrico, incluyendo tests de estacionariedad (ADF), diferenciación de series, selección de lags óptimos (AIC) y pronóstico fuera de muestra para el PIB, consumo e inversión.

### 🚀 **Resultados y Hallazgos Principales**
- **Elastic Net:** Demostró un **rendimiento moderado (R² de 57.7%)** pero una **alta interpretabilidad**, identificando el ingreso medio y la ubicación geográfica como los principales impulsores de los precios de las viviendas.
- **Regresión Cuantílica:** Fue exitosa al modelar la distribución de las horas trabajadas, generando un **intervalo de predicción coherente** (ej. 28-52 horas semanales). Demostró ser superior a una regresión estándar para entender la variabilidad de los resultados.
- **Modelo VAR:** Capturó eficazmente la **tendencia de crecimiento a largo plazo** de los indicadores económicos. Sin embargo, se mostró **incapaz de predecir la volatilidad y los ciclos de corto plazo**, generando pronósticos excesivamente suaves y optimistas.

### 🏆 **Conclusión Final**
El análisis subraya que no existe un "mejor modelo" de regresión, sino una **técnica adecuada para cada pregunta específica**.
- **Elastic Net** es robusto para estimaciones puntuales e interpretables.
- **Regresión Cuantílica** es ideal para cuantificar la incertidumbre y el riesgo.
- **VAR** es la herramienta correcta para analizar tendencias en sistemas de variables interdependientes, aunque con limitaciones para pronósticos tácticos a corto plazo.

# [**Ir al Proyecto**](../Actividad_2_Comparativa_Regresion/Actividad_2_Modulo_5_Aplicación_Comparativa_Técnicas_Avanzadas_Regresión.ipynb)

---

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**
2.  **Clona o descarga este repositorio.**
3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
    ```
4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_2_Modulo_5_Aplicación_Comparativa_Técnicas_Avanzadas_Regresión.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos Avanzados](../README.md) | [Actividad Siguiente →](../Actividad_3_Boosting_Bagging/README.md)
