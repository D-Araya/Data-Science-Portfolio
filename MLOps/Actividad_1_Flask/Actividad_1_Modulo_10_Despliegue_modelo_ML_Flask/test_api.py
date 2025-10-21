# -----------------------------------------------------------------------------
# test_api.py
# -----------------------------------------------------------------------------
# Objetivo:
# Este script prueba el endpoint /predict de la API de Flask para verificar
# su correcto funcionamiento. Envía diferentes conjuntos de datos, incluyendo
# casos válidos e inválidos, y muestra la respuesta del servidor.
#
# Requisitos:
# - La librería 'requests' debe estar instalada (pip install requests).
# - La API de Flask (app.py) debe estar en ejecución.
# -----------------------------------------------------------------------------

# 1. Importación de librerías
import requests # Para realizar peticiones HTTP
import json # Para manejar y mostrar datos en formato JSON

# 2. Configuración de la URL del endpoint
# La URL base donde se está ejecutando la API de Flask.
# Por defecto, Flask se ejecuta en el puerto 5000.
API_URL = "http://127.0.0.1:5000/predict"

# 3. Definición de casos de prueba
# Se define una lista de diccionarios, cada uno representando un caso de prueba.
test_cases = [
    {
        "description": "Caso 1: Predicción válida (espera 'setosa')",
        "payload": {"features": [5.1, 3.5, 1.4, 0.2]},
        "is_valid": True
    },
    {
        "description": "Caso 2: Predicción válida (espera 'versicolor')",
        "payload": {"features": [5.9, 3.0, 4.2, 1.5]},
        "is_valid": True
    },
    {
        "description": "Caso 3: Predicción válida (espera 'virginica')",
        "payload": {"features": [6.9, 3.1, 5.4, 2.1]},
        "is_valid": True
    },
    {
        "description": "Caso 4: Error - Número incorrecto de características",
        "payload": {"features": [5.1, 3.5, 1.4]},
        "is_valid": False
    },
    {
        "description": "Caso 5: Error - Tipo de dato no numérico",
        "payload": {"features": [5.1, "texto", 1.4, 0.2]},
        "is_valid": False
    },
    {
        "description": "Caso 6: Error - 'features' no es una lista",
        "payload": {"features": "no es una lista"},
        "is_valid": False
    },
    {
        "description": "Caso 7: Error - Falta la clave 'features'",
        "payload": {"datos": [5.1, 3.5, 1.4, 0.2]},
        "is_valid": False
    },
    {
        "description": "Caso 8: Error - JSON malformado (simulado como un diccionario vacío)",
        "payload": {},
        "is_valid": False
    }
]

# 4. Ejecución de las pruebas
print("--- Iniciando pruebas de la API ---")

# Iteramos sobre cada caso de prueba definido
for case in test_cases:
    print(f"\n--- {case['description']} ---")
    
    # Imprimimos el JSON que se va a enviar
    print(f"Enviando Payload: {json.dumps(case['payload'])}")
    
    try:
        # Realizamos la petición POST a la API, enviando el payload en formato JSON.
        response = requests.post(API_URL, json=case['payload'])
        
        # Obtenemos el código de estado de la respuesta (e.g., 200, 400, 500)
        status_code = response.status_code
        print(f"Código de Estado Recibido: {status_code}")
        
        # Imprimimos la respuesta JSON del servidor
        print(f"Respuesta del Servidor: {response.json()}")

        # Verificamos si el resultado fue el esperado
        if case['is_valid'] and status_code == 200:
            print("Resultado: ÉXITO (Respuesta válida para datos válidos)")
        elif not case['is_valid'] and status_code == 400:
            print("Resultado: ÉXITO (Error esperado para datos inválidos)")
        else:
            print(f"Resultado: FALLO (Código de estado inesperado: {status_code})")

    except requests.exceptions.ConnectionError as e:
        # Capturamos el error si la API no está en ejecución.
        print("\nERROR: No se pudo conectar a la API.")
        print("Asegúrate de que el servidor Flask (app.py) esté en ejecución.")
        break # Detenemos las pruebas si no hay conexión
    except Exception as e:
        # Capturamos cualquier otro error inesperado.
        print(f"\nOcurrió un error inesperado durante la prueba: {e}")

print("\n--- Pruebas finalizadas ---")