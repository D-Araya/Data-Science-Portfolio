# Cómo Poner Todo en Marcha

Para ejecutar y probar la solución completa, sigue estos pasos en tu terminal.

## 1. Crea y Activa un Entorno Virtual (Recomendado)
Un entorno virtual aísla las dependencias de tu proyecto para evitar conflictos con otros proyectos de Python.

### Crear el entorno virtual
```bash
python -m venv venv
```
version de python 3.11 para este proytecto
```bash
py -3.11 -m venv venv
```


### Activar en macOS/Linux
```bash
source venv/bin/activate
```

### Activar en Windows (PowerShell/CMD)
```bash
venv\Scripts\activate
```

## 2. Instala las Dependencias
Crea un archivo llamado `requirements.txt` con el contenido proporcionado anteriormente y luego instala todas las librerías necesarias con un solo comando.

```bash
pip install -r requirements.txt
```

## 3. Entrena y Guarda el Modelo
Ejecuta el script de entrenamiento. Este paso creará el archivo `modelo.pkl` que la API necesita para funcionar.

```bash
python train_model.py
```

Verás en la consola los mensajes confirmando que el modelo fue entrenado y guardado correctamente.

## 4. Inicia la API con Flask
Levanta el servidor web que expondrá tu modelo.

```bash
python app.py
```

La terminal te indicará que el servidor está corriendo en `http://127.0.0.1:5000`. Debes dejar esta terminal abierta para que la API se mantenga activa.

## 5. Prueba la API
Abre una nueva ventana de terminal (sin cerrar la que ejecuta el servidor) y lanza el script de prueba.

```bash
python test_api.py
```

Este script enviará múltiples solicitudes (válidas e inválidas) a la API y mostrará en la consola los resultados de cada prueba, permitiendo verificar que todo funciona como se espera.

---

[Volver al índice principal](../../README.md) | [Volver a MLOps](../README.md) | [Actividad Siguiente →](../Actividad_2_Docker/README.md)