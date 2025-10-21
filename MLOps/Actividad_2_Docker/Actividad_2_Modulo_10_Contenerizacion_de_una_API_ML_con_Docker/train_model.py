# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------
# train_model.py
# ----------------------------------------------------------------------------------
#
# Descripción:
# Este script se encarga de entrenar un modelo de Machine Learning y guardarlo
# en un archivo para su posterior uso en una API.
#
# Pasos:
# 1. Cargar un conjunto de datos (en este caso, el dataset "Wine" de scikit-learn).
# 2. Dividir los datos en conjuntos de entrenamiento y prueba.
# 3. Inicializar y entrenar un modelo de clasificación (RandomForestClassifier).
# 4. Guardar el modelo entrenado en un archivo .pkl usando joblib.
#
# ----------------------------------------------------------------------------------

# 1. Importación de librerías necesarias
import pandas as pd  # Para la manipulación de datos, aunque aquí se usa principalmente para la introspección.
from sklearn.model_selection import train_test_split  # Para dividir los datos.
from sklearn.ensemble import RandomForestClassifier  # El modelo de Machine Learning que usaremos.
from sklearn.datasets import load_wine  # El dataset de ejemplo.
import joblib  # Para guardar y cargar el modelo entrenado.

# 2. Carga del conjunto de datos
# Se carga el dataset "Wine", que contiene características de vinos y su clasificación.
print("Cargando el dataset de vinos...")
wine = load_wine()
X, y = wine.data, wine.target

# 3. División de los datos
# Se dividen los datos en un 80% para entrenamiento y un 20% para pruebas.
# `random_state` se usa para garantizar que la división sea siempre la misma, lo que permite la reproducibilidad.
print("Dividiendo los datos en conjuntos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inicialización del modelo
# Se crea una instancia del clasificador de Bosque Aleatorio (Random Forest).
# `n_estimators` define el número de árboles en el bosque.
# `random_state` para la reproducibilidad del modelo.
print("Inicializando el modelo RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Entrenamiento del modelo
# Se ajusta el modelo a los datos de entrenamiento (características y etiquetas).
print("Entrenando el modelo...")
model.fit(X_train, y_train)

# 6. Guardado del modelo entrenado
# Se utiliza joblib.dump para serializar y guardar el objeto del modelo en un archivo.
# Este archivo 'modelo.pkl' contendrá todo el conocimiento que el modelo ha aprendido.
# El protocolo de compresión '3' es eficiente.
print("Guardando el modelo entrenado en 'modelo.pkl'...")
joblib.dump(model, 'modelo.pkl', compress=3)

print("\n¡Entrenamiento completado y modelo guardado exitosamente!")
print(f"El modelo tiene una precisión de {model.score(X_test, y_test):.2f} en el conjunto de prueba.")
