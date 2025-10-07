# Segmentación de Clientes con Clustering Jerárquico y PCA

Este proyecto utiliza técnicas de aprendizaje no supervisado para descubrir la estructura oculta en datos de clientes de un centro comercial. Se combinan el clustering jerárquico para agrupar a los clientes y el Análisis de Componentes Principales (PCA) para visualizar los segmentos resultantes.

---

### 🎯 **Objetivo Principal**
- **Propósito:** Aplicar clustering jerárquico para identificar grupos naturales y distintos dentro de un conjunto de datos de clientes, sin etiquetas predefinidas.
- **Problema de Negocio:** Segmentar a los clientes de un centro comercial en perfiles claros basados en su **ingreso anual** y su **puntaje de gasto**. El objetivo es proporcionar a los equipos de marketing una base sólida para diseñar campañas personalizadas y estrategias dirigidas.

### 📊 **Dataset Utilizado**
- **Nombre:** `Mall_Customers.csv`
- **Fuente:** Un conjunto de datos que contiene información demográfica y de comportamiento de compra de clientes de un centro comercial, cargado desde un archivo CSV local.

### 🛠️ **Metodología y Modelos Aplicados**
El análisis sigue un flujo de trabajo de aprendizaje no supervisado bien definido:
1.  **Preprocesamiento de Datos:** Se seleccionaron las características clave (`Annual Income` y `Spending Score`) y se estandarizaron con `StandardScaler` para asegurar que el cálculo de distancias no se viera sesgado por las diferentes escalas.
2.  **Análisis de Dendrograma:** Se generó un dendrograma utilizando el método de enlace de 'ward' para visualizar la estructura jerárquica de los datos. Este análisis fue crucial para determinar que **5 clusters** era el número óptimo de segmentos.
3.  **Clustering Jerárquico Aglomerativo:** Se aplicó el modelo `AgglomerativeClustering` para agrupar a los clientes en los 5 segmentos identificados.
4.  **Reducción de Dimensionalidad para Visualización:** Se utilizó el Análisis de Componentes Principales (PCA) para transformar los datos a un espacio de dos dimensiones, permitiendo una visualización clara de los clusters en un gráfico de dispersión.

### 🚀 **Resultados y Hallazgos Principales**
- **Segmentación Exitosa:** Se identificaron **5 perfiles de clientes** claros, distintos y accionables, basados en sus patrones de ingreso y gasto.
- **Perfiles de Clientes Identificados 🏆:** Los grupos encontrados son:
    1.  **Ingresos Altos, Gasto Bajo** (Cautelosos)
    2.  **Ingresos Bajos, Gasto Bajo** (Promedio/Estándar)
    3.  **Ingresos Medios, Gasto Medio** (Público Objetivo Principal)
    4.  **Ingresos Bajos, Gasto Alto** (Jóvenes/Promociones)
    5.  **Ingresos Altos, Gasto Alto** (Clientes VIP/Ideales)
- **Visualización Clara:** El uso combinado de PCA y `matplotlib` permitió una visualización nítida de los segmentos, validando visualmente la cohesión interna de cada cluster y su separación respecto a los demás.

### 🏆 **Recomendación Final**
- La combinación de **Clustering Jerárquico** (para encontrar grupos) y **PCA** (para visualizar) es una técnica robusta y altamente efectiva para tareas de segmentación de clientes.
- Los 5 segmentos descubiertos proporcionan insights valiosos y deben ser utilizados por el equipo de marketing para desarrollar campañas personalizadas, mejorar la retención y optimizar la asignación de recursos.


# [**Ir al Proyecto**](../Actividad_1_Clustering_Jerarquico/Actividad_1_Modulo_6_Explorando_estructura_oculta_de_datos_con_clustering_jerárquico_y_reducción_de_dimensionalidad.ipynb)


---

## ⚙️ **Cómo Ejecutar el Notebook**

Para correr este proyecto en tu entorno local, sigue estos pasos:

1.  **Asegúrate de tener Python 3.8 o superior.**
2.  **Clona o descarga este repositorio.** El archivo `Mall_Customers.csv` debe estar en el mismo directorio que el notebook.
3.  **Instala las librerías necesarias:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
4.  **Ejecuta el notebook de Jupyter:**
    `Actividad_1_Modulo_6_Explorando_estructura_oculta_de_datos_con_clustering_jerárquico_y_reducción_de_dimensionalidad.ipynb`

---

[Volver al índice principal](../../README.md) | [Volver a Modelos No Supervisados](../README.md) | [Actividad Siguiente →](../Actividad_2_DBSCAN_HDBSCAN/README.md)
