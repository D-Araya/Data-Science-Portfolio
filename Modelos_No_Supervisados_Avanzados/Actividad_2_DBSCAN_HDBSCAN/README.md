# Clustering Basado en Densidad con DBSCAN y HDBSCAN

Este proyecto realiza un análisis comparativo de dos algoritmos de clustering avanzados basados en densidad: DBSCAN y HDBSCAN. El objetivo es demostrar su capacidad para identificar grupos de formas no lineales (arbitrarias) y su habilidad para detectar y aislar puntos de ruido (outliers) en un conjunto de datos.

---

### 🎯 **Objetivo Principal**
- **Propósito:** Aplicar, visualizar y comparar el funcionamiento de los algoritmos de clustering DBSCAN y HDBSCAN en un dataset diseñado para desafiar a los métodos tradicionales.
- **Problema de Negocio:** Mostrar cómo estos algoritmos pueden ser utilizados para tareas de segmentación donde los grupos no son esféricos y es necesario identificar datos anómalos de forma automática, como en la detección de fraudes o la segmentación de comportamiento de usuarios.

### 📊 **Dataset Utilizado**
- **Nombre:** `make_moons`
- **Fuente:** Un dataset sintético generado con la librería `sklearn.datasets`. Se diseñó con dos clusters en forma de "media luna" y se le añadió ruido para simular un escenario realista y complejo.

### 🛠️ **Metodología y Modelos Aplicados**
El análisis se centra en la aplicación directa y la comparación de dos modelos de clustering:
1.  **Preprocesamiento:** Los datos generados se normalizaron utilizando `StandardScaler` para asegurar que las métricas de distancia no se vieran afectadas por la escala de las características.
2.  **Modelos de Clustering por Densidad:**
    - **`DBSCAN` (Density-Based Spatial Clustering of Applications with Noise):** Se ajustó el modelo configurando manualmente los parámetros clave `eps` (radio de vecindad) y `min_samples` (puntos mínimos por cluster).
    - **`HDBSCAN` (Hierarchical DBSCAN):** Se aplicó este algoritmo más avanzado, que es menos sensible a los parámetros y solo requiere definir el `min_cluster_size`.
3.  **Visualización y Evaluación:** Los resultados de ambos modelos se representaron en gráficos de dispersión para evaluar visualmente su capacidad para diferenciar los clusters y clasificar correctamente el ruido.

### 🚀 **Resultados y Hallazgos Principales**
- **Identificación Exitosa de Clusters No Lineales:** Tanto **DBSCAN** como **HDBSCAN** lograron identificar perfectamente los dos clusters con forma de media luna, demostrando su superioridad sobre algoritmos como K-Means para datos con estructuras de forma arbitraria.
- **Excelente Detección de Ruido 🏆:** La principal fortaleza destacada fue la habilidad de ambos modelos para **aislar los puntos de ruido**, asignándoles correctamente la etiqueta de outlier (-1) y separándolos de los clusters densos y cohesivos.
- **HDBSCAN como Opción Más Robusta:** Aunque ambos modelos tuvieron éxito, **HDBSCAN** se perfila como una alternativa más moderna y robusta. Su principal ventaja es que no requiere el ajuste del parámetro `eps`, lo que lo hace más fácil de usar y más adaptable a clusters con densidades variables.

### 🏆 **Recomendación Final**
- Para tareas de clustering donde se sospecha la existencia de grupos con formas complejas o la presencia de datos anómalos, los algoritmos basados en densidad como **DBSCAN** y **HDBSCAN** son la elección recomendada.
- Se sugiere utilizar **HDBSCAN** como primera opción debido a su mayor facilidad de uso y robustez, ya que reduce la necesidad de una experimentación exhaustiva con los hiperparámetros.

# [**Ir al Proyecto**](../Actividad_2_DBSCAN_HDBSCAN/Actividad_2_Modulo_6_Clustering_con_DBSCAN_y_DBHSCAN.ipynb)

---

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**
2.  **Clona o descarga este repositorio.**
3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn hdbscan
    ```
4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_2_Modulo_6_Clustering_con_DBSCAN_y_DBHSCAN.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos No Supervisados](../README.md) | [Actividad Siguiente →](../Actividad_3_PCA/README.md)
