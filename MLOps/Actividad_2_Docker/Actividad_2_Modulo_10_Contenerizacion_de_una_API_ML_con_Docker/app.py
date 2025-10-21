# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------
# app.py
# ----------------------------------------------------------------------------------
#
# Descripción:
# Este script crea una API REST utilizando Flask para servir el modelo de ML entrenado.
# La API expone dos endpoints:
#   - '/': Devuelve un mensaje de bienvenida.
#   - '/predict': Recibe datos de un vino en formato JSON y devuelve la clase predicha.
#
# ----------------------------------------------------------------------------------

# 1. Importación de librerías necesarias
from flask import Flask, request, jsonify  # Flask para crear la API, request para manejar las peticiones, jsonify para formatear respuestas JSON.
import joblib  # Para cargar el modelo .pkl.
import numpy as np  # Para realizar operaciones numéricas, especialmente para convertir la entrada a un array.
from sklearn.datasets import load_wine # Para obtener los nombres de las clases de vino.

# 2. Inicialización de la aplicación Flask
app = Flask(__name__)

# 3. Carga del modelo entrenado
# Se carga el archivo .pkl que contiene el modelo de clasificación de vinos.
# Es importante que este archivo esté en el mismo directorio o se especifique la ruta correcta.
print("Cargando el modelo desde 'modelo.pkl'...")
try:
    model = joblib.load('modelo.pkl')
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'modelo.pkl' no fue encontrado. Asegúrate de entrenar el modelo primero.")
    model = None

# 4. Carga de los nombres de las clases del dataset
# Esto se hace para que la respuesta de la API sea más descriptiva (e.g., "class_0")
# en lugar de solo un número (0, 1, 2).
wine_dataset = load_wine()
target_names = wine_dataset.target_names

# 5. Definición del endpoint raíz ('/')
# Este endpoint es el punto de entrada principal de la API.
@app.route('/', methods=['GET'])
def welcome():
    # Devuelve un mensaje simple en formato JSON para indicar que la API está funcionando.
    return jsonify({'message': 'Bienvenido a la API de Clasificacion de Vinos. Usa el endpoint /predict para obtener una prediccion.'})

# 6. Definición del endpoint de predicción ('/predict')
# Este endpoint acepta peticiones POST con datos en formato JSON.
@app.route('/predict', methods=['POST'])
def predict():
    # Primero, se verifica si el modelo fue cargado correctamente.
    if model is None:
        return jsonify({'error': 'Modelo no disponible. Revisa los logs del servidor.'}), 500

    try:
        # Se obtienen los datos JSON de la petición. `force=True` evita errores de content-type.
        json_data = request.get_json(force=True)

        # Se extrae la lista de características (features) del JSON.
        features = json_data['features']

        # Se convierten las características a un array de NumPy y se le da la forma correcta.
        # El modelo espera un array 2D, por eso se usa reshape(1, -1).
        final_features = np.array(features).reshape(1, -1)

        # Se realiza la predicción utilizando el modelo cargado.
        prediction_index = model.predict(final_features)

        # Se obtiene el nombre de la clase predicha usando el índice devuelto por el modelo.
        predicted_class_name = target_names[prediction_index[0]]

        # Se devuelve el resultado de la predicción en formato JSON.
        return jsonify({'prediction': predicted_class_name})

    except Exception as e:
        # Si ocurre algún error durante el proceso (e.g., JSON mal formado), se devuelve un mensaje de error.
        return jsonify({'error': str(e)}), 400

# 7. Punto de entrada para ejecutar la aplicación
# Este bloque se ejecuta solo si el script es llamado directamente (no importado).
if __name__ == '__main__':
    # Se inicia el servidor de desarrollo de Flask.
    # host='0.0.0.0' hace que sea accesible desde fuera del contenedor Docker.
    # port=5000 es el puerto estándar para Flask.
    # debug=True activa el modo de depuración para ver errores detallados.
    app.run(host='0.0.0.0', port=5000, debug=True)