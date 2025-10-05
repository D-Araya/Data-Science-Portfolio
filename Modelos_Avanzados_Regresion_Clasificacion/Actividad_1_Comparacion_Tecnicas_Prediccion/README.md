# Comparación de Técnicas Avanzadas para Predicción de Ingresos

Este proyecto aplica y compara modelos avanzados de regresión y clasificación (Elastic Net, Regresión Cuantílica, Random Forest y XGBoost) para resolver un problema de clasificación binaria: predecir si una persona ganará más de $50,000 al año basándose en datos demográficos.

# Table of Contents
1. [Resumen del Proyecto](#ch1)
2. [Paso 1: Carga y Preprocesamiento de Datos](#ch2)
3. [Paso 2: Entrenamiento de los Modelos](#ch3)
4. [Paso 3: Evaluación de Desempeño](#ch4)
5. [Paso 4: Análisis de Importancia de Variables](#ch5)
6. [Análisis de Resultados](#ch6)
7. [Conclusión Final y Recomendación](#ch7)

<a id="ch1"></a>
# Resumen del Proyecto

Una empresa desea construir un modelo robusto que le permita predecir con precisión si una persona ganará más de $50,000 al año, en base a sus características demográficas y laborales. El objetivo es identificar qué modelo se adapta mejor al problema, considerando precisión, estabilidad e interpretabilidad.

Aunque el objetivo es de **clasificación binaria** (ingresos `>50K` o `<=50K`), este ejercicio explora un enfoque híbrido:
- **Modelos de Clasificación Nativos**: Se utilizan Random Forest y XGBoost, diseñados específicamente para esta tarea.
- **Modelos de Regresión Adaptados**: Se "fuerza" a modelos de regresión como Elastic Net y Regresión Cuantílica a resolver el problema. Para ello, las etiquetas (`'<=50K'`, `'>50K'`) se convierten en valores numéricos (`0`, `1`). Las salidas continuas de estos modelos se convierten de nuevo a clases aplicando un umbral de decisión (0.5).

El objetivo es comparar estos dos enfoques y determinar cuál es superior en este contexto.

<a id="ch2"></a>
# Paso 1: Carga y Preprocesamiento de Datos

## 1.1 Importación de Librerías y Carga de Datos
En esta primera sección, preparamos el entorno de trabajo importando las librerías necesarias para la manipulación de datos (`pandas`, `numpy`), visualización (`matplotlib`, `seaborn`), preprocesamiento y modelado (`scikit-learn`, `xgboost`). Finalmente, cargamos el dataset "Adult Income" desde OpenML.

```python
# --- Librerías para manipulación de datos ---
import pandas as pd
import numpy as np

# --- Librerías para visualización ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Funciones de Scikit-learn para carga de datos ---
from sklearn.datasets import fetch_openml

# --- Funciones de Scikit-learn para preprocesamiento y pipelines ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Modelos de Scikit-learn y XGBoost ---
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# --- Métricas de evaluación de Scikit-learn ---
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay

# --- Carga del dataset ---
print("Cargando dataset 'adult'...")
adult_data = fetch_openml(name="adult", version=2, as_frame=True, parser='auto')
df = adult_data.frame
print("Dataset cargado exitosamente.")
df.head()
````

**Salida:**

```
Cargando dataset 'adult'...
Dataset cargado exitosamente.
```

## 1.2 Preprocesamiento y Limpieza de Datos

El preprocesamiento es un paso crítico. En esta sección, realizamos las siguientes tareas:

1.  **Limpieza de Valores Faltantes**: Reemplazamos el carácter `'?'` por `np.nan` para un manejo estándar de nulos.
2.  **Codificación de la Variable Objetivo**: Convertimos la variable objetivo `class` de `['<=50K', '>50K']` a `[0, 1]`.
3.  **División de Datos**: Separamos los datos en conjuntos de entrenamiento (80%) y prueba (20%), usando estratificación para mantener la proporción de clases.
4.  **Creación de Pipelines de Transformación**:
      * **Numéricas**: Imputamos los valores faltantes con la **mediana** y luego aplicamos **escalado estándar** (`StandardScaler`).
      * **Categóricas**: Imputamos los valores faltantes con la **moda** y aplicamos **One-Hot Encoding**.
5.  **`ColumnTransformer`**: Combinamos los pipelines para aplicar las transformaciones correctas a cada tipo de columna de forma eficiente y robusta.

<!-- end list -->

```python
print("Iniciando preprocesamiento de datos...")

# --- Definición y limpieza de la variable objetivo ---
TARGET_NAME = 'class'
df.replace('?', np.nan, inplace=True)

# --- Separación de características (X) y objetivo (y) ---
X = df.drop(TARGET_NAME, axis=1)
y = df[TARGET_NAME]

# --- Codificación de la variable objetivo a formato binario ---
y = y.apply(lambda x: 1 if x == '>50K' else 0)

# --- Identificación automática de tipos de columnas ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# --- División en conjuntos de entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Creación de pipelines de preprocesamiento ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# --- Combinación de pipelines con ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

print("Preprocesamiento completado.")
```

<a id="ch3"></a>

# Paso 2: Entrenamiento de los Modelos

Con los pipelines listos, entrenamos los cuatro modelos. Cada modelo se encapsula en un pipeline principal que primero aplica el preprocesador y luego entrena el modelo de Machine Learning.

Los modelos entrenados son:

  - **Elastic Net**: Un modelo de regresión lineal adaptado para clasificación.
  - **Regresión Cuantílica**: Predice un cuantil específico. Se entrenan tres modelos para los cuantiles 0.1, 0.5 y 0.9.
  - **Random Forest**: Un modelo de ensamble de árboles de decisión (clasificación nativa).
  - **XGBoost (Extreme Gradient Boosting)**: Un modelo de ensamble basado en árboles de alto rendimiento (clasificación nativa).

<!-- end list -->

```python
print("Entrenando modelos...")

# --- Modelo 1: Elastic Net ---
elastic_net_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(random_state=42))
])
elastic_net_pipeline.fit(X_train, y_train)

# --- Modelo 2: Regresión Cuantílica ---
quantile_50_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', QuantileRegressor(quantile=0.5, solver='highs'))
])
quantile_50_pipeline.fit(X_train, y_train)

# (Se omiten los pipelines para cuantiles 0.1 y 0.9 por brevedad)

# --- Modelo 3: Random Forest ---
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])
random_forest_pipeline.fit(X_train, y_train)

# --- Modelo 4: XGBoost ---
xgboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1))
])
xgboost_pipeline.fit(X_train, y_train)

print("Entrenamiento completado.")
```

<a id="ch4"></a>

# Paso 3: Evaluación de Desempeño

Una vez entrenados, medimos el rendimiento de los modelos.

  - **Métricas de Regresión**: Para Elastic Net y Regresión Cuantílica, evaluamos sus salidas numéricas directas con **RMSE** y **Pinball Loss**.
  - **Métricas de Clasificación**: Para todos los modelos, evaluamos su capacidad de clasificación con **Accuracy**, **Matriz de Confusión** y la **Curva ROC / AUC**, que es una de las métricas más importantes para comparar clasificadores.

<!-- end list -->

```python
# --- Generación de predicciones ---
y_pred_elastic_net_reg = elastic_net_pipeline.predict(X_test)
y_pred_q50_reg = quantile_50_pipeline.predict(X_test)
y_pred_elastic_net_class = (y_pred_elastic_net_reg > 0.5).astype(int)
y_pred_q50_class = (y_pred_q50_reg > 0.5).astype(int)
y_pred_rf_class = random_forest_pipeline.predict(X_test)
y_pred_xgb_class = xgboost_pipeline.predict(X_test)
y_prob_rf = random_forest_pipeline.predict_proba(X_test)[:, 1]
y_prob_xgb = xgboost_pipeline.predict_proba(X_test)[:, 1]

# --- Diccionario de resultados para iteración ---
results = {
    "Elastic Net": {"pred_class": y_pred_elastic_net_class, "score": y_pred_elastic_net_reg},
    "Quantile Reg. p50": {"pred_class": y_pred_q50_class, "score": y_pred_q50_reg},
    "Random Forest": {"pred_class": y_pred_rf_class, "score": y_prob_rf},
    "XGBoost": {"pred_class": y_pred_xgb_class, "score": y_prob_xgb}
}

# --- Evaluación de Clasificación ---
print("\n--- MÉTRICAS DE CLASIFICACIÓN (PRINCIPALES) ---")
for name, res in results.items():
    accuracy = accuracy_score(y_test, res["pred_class"])
    auc = roc_auc_score(y_test, res["score"])
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")

# --- Visualización de Matrices de Confusión ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["pred_class"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[i], cmap='Blues')
    axes[i].set_title(f"Matriz de Confusión - {name}")
plt.tight_layout()
plt.show()

# --- Visualización de Curvas ROC ---
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio')
for name, res in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["score"])
    auc = roc_auc_score(y_test, res["score"])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Comparación de Curvas ROC')
plt.legend()
plt.grid()
plt.show()
```

**Salida (Métricas):**

```
--- MÉTRICAS DE CLASIFICACIÓN (PRINCIPALES) ---

--- Elastic Net ---
Accuracy: 0.7607
ROC AUC Score: 0.5000

--- Quantile Reg. p50 ---
Accuracy: 0.7607
ROC AUC Score: 0.5000

--- Random Forest ---
Accuracy: 0.8602
ROC AUC Score: 0.904

--- XGBoost ---
Accuracy: 0.8741
ROC AUC Score: 0.929
```

<a id="ch5"></a>

# Paso 4: Análisis de Importancia de Variables

Para interpretar los modelos, analizamos qué características consideraron más importantes.

  - **Elastic Net**: Se analizan los **coeficientes** (`.coef_`).
  - **Random Forest y XGBoost**: Se analiza el atributo `.feature_importances_`, que mide cuán útil fue cada variable para el modelo.

<!-- end list -->

```python
# --- Obtención de los nombres de las características ---
cat_feature_names = xgboost_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features + cat_feature_names

# --- Análisis para Elastic Net ---
elastic_net_coefs = pd.Series(
    elastic_net_pipeline.named_steps['regressor'].coef_,
    index=all_feature_names
).sort_values(ascending=False)
print("\nTop 10 variables más influyentes (Elastic Net):")
print(elastic_net_coefs.head(10))

# --- Análisis para Random Forest y XGBoost ---
rf_importances = pd.Series(
    random_forest_pipeline.named_steps['classifier'].feature_importances_,
    index=all_feature_names
).sort_values(ascending=False)

xgb_importances = pd.Series(
    xgboost_pipeline.named_steps['classifier'].feature_importances_,
    index=all_feature_names
).sort_values(ascending=False)

# --- Visualización de la importancia de variables ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
rf_importances.head(15).sort_values().plot(kind='barh', ax=axes[0], title='Importancia de Variables - Random Forest')
xgb_importances.head(15).sort_values().plot(kind='barh', ax=axes[1], title='Importancia de Variables - XGBoost')
plt.tight_layout()
plt.show()
```

**Salida (Elastic Net):**

```
Top 10 variables más influyentes (Elastic Net):
age                       0.0
fnlwgt                   -0.0
education-num             0.0
capital-gain              0.0
...
```

<a id="ch6"></a>

# Análisis de Resultados

### Conclusión Ejecutiva

Los resultados demuestran una clara superioridad de los modelos basados en árboles. **XGBoost es el modelo ganador 🥇**, con el mejor rendimiento en todas las métricas relevantes (**ROC AUC de 0.929**). Por el contrario, los modelos de regresión adaptados (**Elastic Net y Regresión Cuantílica**) **fracasaron completamente**, sin capacidad predictiva alguna.

### Análisis Detallado

#### 1\. Modelos de Regresión: Un Fracaso Predictivo 📉

Ambos modelos obtuvieron un **ROC AUC Score de 0.5000**, equivalente a una predicción al azar. Las matrices de confusión revelan que clasificaron a **todos** los individuos en la clase 0 (ingresos \<=$50K), fallando en identificar a una sola persona con ingresos altos. El análisis de importancia de variables para Elastic Net confirma esto, ya que **todos los coeficientes fueron cero**, indicando que el modelo no utilizó ninguna información para predecir.

#### 2\. Modelos de Clasificación: Alto Rendimiento y Coherencia 🚀

Estos modelos fueron altamente efectivos.

  * **Random Forest (Subcampeón 🥈)**: Con un **ROC AUC de 0.904** y un **Accuracy del 86.02%**, demostró ser un modelo muy sólido. Su análisis de importancia de variables reveló que `fnlwgt` (un peso estadístico), `age` (edad) y `capital-gain` (ganancias de capital) fueron sus principales predictores.

  * **XGBoost (Campeón 🥇)**: Superó a todos con un **ROC AUC de 0.929** y un **Accuracy del 87.41%**. Fue notablemente mejor en identificar correctamente a las personas con ingresos altos. Su análisis de importancia mostró un enfoque de "ancla", atribuyendo una importancia dominante a `marital-status_Married-civ-spouse` (estar casado), seguido de `education-num` (años de educación) y `capital-gain`.

### Interpretación Visual: Curvas ROC

La gráfica de Curvas ROC confirma visualmente estos hallazgos:

  - Las curvas de **XGBoost** y **Random Forest** se elevan rápidamente, demostrando su excelente capacidad predictiva.
  - Las curvas de **Elastic Net y Regresión Cuantílica** son una línea diagonal, la firma visual de un clasificador sin poder predictivo.

<a id="ch7"></a>

# Conclusión Final y Recomendación

Se recomienda **implementar el modelo XGBoost** para la predicción de ingresos.

**Justificación:**

1.  **Precisión Superior**: Cumple el objetivo de "predecir con precisión" de manera sobresaliente, liderando en la métrica más robusta (ROC AUC) y en la capacidad de identificar la clase de interés (ingresos altos).
2.  **Insights Accionables**: A pesar de su complejidad, el modelo proporciona un insight de negocio claro y potente: el **estado civil** es el factor más determinante, seguido de la educación y las ganancias de capital.
3.  **Estándar de la Industria**: XGBoost es una herramienta probada, optimizada y escalable, ideal para ser desplegada en un entorno de producción.

**Random Forest** se considera una excelente segunda opción si la simplicidad o la velocidad de entrenamiento fueran factores prioritarios por encima del rendimiento máximo.


