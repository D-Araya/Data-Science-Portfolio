# -----------------------------------------------------------------------------
# app.py
# -----------------------------------------------------------------------------
# Objetivo:
# Implementar una API REST con Flask para servir predicciones de un modelo
# de Machine Learning previamente entrenado y guardado en 'modelo.pkl'.
#
# Endpoints:
# - GET /: Devuelve un mensaje de bienvenida para verificar que la API está activa.
# - POST /predict: Recibe datos en formato JSON, realiza una predicción y
#   devuelve el resultado.
#
# Características:
# - Carga de modelo serializado.
# - Validación de datos de entrada.
# - Manejo de errores (códigos HTTP 400 y 500).
# - Habilitación de CORS para permitir peticiones desde otros dominios.
# -----------------------------------------------------------------------------

# 1. Importación de librerías
from flask import Flask, request, jsonify # Flask para la API, request para acceder a los datos, jsonify para formatear respuestas
from flask_cors import CORS # Para permitir peticiones Cross-Origin
import joblib # Para cargar el modelo guardado
import numpy as np # Para manipulación de arrays numéricos
from sklearn.datasets import load_iris # Para obtener los nombres de las clases

# 2. Inicialización de la aplicación Flask
app = Flask(__name__)
# Habilitamos CORS para que la API pueda ser consumida desde cualquier origen.
# Esto es útil si tienes un frontend en un dominio diferente.
CORS(app)

# 3. Carga del modelo y datos adicionales
try:
    # Cargamos el modelo entrenado desde el archivo 'modelo.pkl'
    model = joblib.load('modelo.pkl')
    # Cargamos el dataset de Iris para obtener los nombres de las clases (target_names)
    # Esto nos permite devolver un nombre de clase legible en lugar de un número.
    iris = load_iris()
    target_names = iris.target_names
    print("Modelo y nombres de las clases cargados correctamente.")
except FileNotFoundError:
    print("Error: El archivo 'modelo.pkl' no fue encontrado. Asegúrate de ejecutar train_model.py primero.")
    model = None
    target_names = None
except Exception as e:
    print(f"Ocurrió un error al cargar el modelo: {e}")
    model = None
    target_names = None

# 4. Definición de Endpoints

# Endpoint Raíz (/)
@app.route('/', methods=['GET'])
def home():
    # Este endpoint es útil para una verificación rápida de que la API está funcionando.
    return jsonify({"message": "API de prediccion de especies de Iris esta activa. Usa el endpoint /predict para predecir."})

# Endpoint de Predicción (/predict)
@app.route('/predict', methods=['POST'])
def predict():
    # Primero, verificamos que el modelo se haya cargado correctamente.
    if model is None or target_names is None:
        # Si el modelo no se cargó, devolvemos un error 500 (Internal Server Error).
        return jsonify({"error": "El modelo no está disponible. Revisa los logs del servidor."}), 500

    try:
        # Obtenemos los datos JSON enviados en la petición POST.
        data = request.get_json()
        if data is None:
            # Si no hay JSON en el cuerpo de la petición, devolvemos un error 400 (Bad Request).
            return jsonify({"error": "Cuerpo de la solicitud no es un JSON válido."}), 400

        # Validación 1: Verificamos que la clave 'features' exista en el JSON.
        if 'features' not in data:
            return jsonify({"error": "La clave 'features' no se encuentra en el JSON."}), 400

        features = data['features']

        # Validación 2: Verificamos que 'features' sea una lista.
        if not isinstance(features, list):
            return jsonify({"error": "'features' debe ser una lista."}), 400
        
        # Validación 3: Verificamos que la lista tenga la cantidad correcta de características (4 para Iris).
        if len(features) != 4:
            return jsonify({"error": f"Se esperaban 4 características, pero se recibieron {len(features)}."}), 400

        # Validación 4: Verificamos que todos los elementos de la lista sean numéricos (int o float).
        if not all(isinstance(f, (int, float)) for f in features):
            return jsonify({"error": "Todas las características deben ser valores numéricos."}), 400
            
        # Convertimos la lista de características a un array de NumPy y la redimensionamos.
        # El modelo espera una entrada 2D, por eso usamos reshape(1, -1).
        final_features = np.array(features).reshape(1, -1)

        # Realizamos la predicción con el modelo cargado.
        prediction_index = model.predict(final_features)
        
        # Obtenemos el nombre de la clase predicha usando el índice devuelto por el modelo.
        predicted_class = target_names[prediction_index[0]]

        # Devolvemos la predicción en formato JSON con un código de estado 200 (OK).
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        # Si ocurre cualquier otro error inesperado durante el proceso,
        # lo capturamos y devolvemos un error 500.
        print(f"Error durante la predicción: {e}")
        return jsonify({"error": "Ocurrió un error interno en el servidor."}), 500

# 5. Punto de entrada para ejecutar la aplicación
if __name__ == '__main__':
    # app.run() inicia el servidor de desarrollo de Flask.
    # debug=True permite ver los errores detallados en el navegador y recarga
    # el servidor automáticamente cuando se hacen cambios en el código.
    # Se recomienda usar un servidor WSGI como Gunicorn para producción.
    app.run(debug=True, port=5000)