# -----------------------------------------------------------------------------
# train_model.py
# -----------------------------------------------------------------------------
# Objetivo:
# Este script entrena un modelo de clasificación de Machine Learning utilizando
# el dataset "Iris" y lo guarda en un archivo para su uso posterior en una API.
#
# Pasos:
# 1. Importar las librerías necesarias.
# 2. Cargar el dataset de Iris.
# 3. Inicializar y entrenar un modelo (RandomForestClassifier).
# 4. Serializar y guardar el modelo entrenado en un archivo 'modelo.pkl'.
# -----------------------------------------------------------------------------

# 1. Importación de librerías
from sklearn.datasets import load_iris # Para cargar el dataset de Iris
from sklearn.ensemble import RandomForestClassifier # Para el modelo de clasificación
import joblib # Para guardar el modelo entrenado

print("Iniciando el entrenamiento del modelo...")

# 2. Cargar el dataset
# El dataset de Iris contiene 150 muestras de flores de Iris, cada una con 4
# características (largo y ancho de sépalo y pétalo) y una de tres especies.
iris = load_iris()
X, y = iris.data, iris.target # X: características, y: etiquetas (especies)

# Se imprime la forma de los datos para verificar
print(f"Dataset cargado. Forma de las características (X): {X.shape}, Forma del objetivo (y): {y.shape}")

# 3. Inicializar y entrenar el modelo
# Se utiliza un RandomForestClassifier, un modelo robusto y popular.
# 'n_estimators=100' significa que se construirán 100 árboles de decisión.
# 'random_state=42' asegura que el resultado sea reproducible.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# El método fit() entrena el modelo con nuestros datos (características y etiquetas)
print("Entrenando el modelo RandomForestClassifier...")
model.fit(X, y)
print("Modelo entrenado exitosamente.")

# 4. Guardar el modelo entrenado
# Utilizamos joblib.dump para serializar el objeto del modelo y guardarlo
# en un archivo llamado 'modelo.pkl'. Este archivo podrá ser cargado
# posteriormente para hacer predicciones sin necesidad de re-entrenar.
joblib.dump(model, 'modelo.pkl')

print("Modelo guardado exitosamente en 'modelo.pkl'.")
print("El script de entrenamiento ha finalizado.")
