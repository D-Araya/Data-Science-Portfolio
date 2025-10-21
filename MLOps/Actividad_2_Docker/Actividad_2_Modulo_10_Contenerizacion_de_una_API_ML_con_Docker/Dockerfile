# ----------------------------------------------------------------------------------
# Dockerfile
# ----------------------------------------------------------------------------------
#
# Descripción:
# Este archivo contiene las instrucciones para construir una imagen de Docker
# que ejecutará la API de Flask. La imagen se basa en una versión de Python,
# instala las dependencias, copia los archivos de la aplicación y define
# el comando para iniciar el servidor.
#
# ----------------------------------------------------------------------------------

# --- ETAPA 1: Definir la imagen base ---
# Se utiliza una imagen oficial de Python 3.11. `3.11-slim` es una versión
# ligera, lo que resulta en una imagen final de menor tamaño y es coherente
# con el entorno de desarrollo.
FROM python:3.11-slim

# --- ETAPA 2: Establecer el directorio de trabajo ---
# Define el directorio de trabajo dentro del contenedor. Todos los comandos
# posteriores (como COPY, RUN) se ejecutarán desde este directorio.
WORKDIR /app

# --- ETAPA 3: Instalar las dependencias ---
# Primero, copia solo el archivo de requerimientos. Docker funciona con un sistema
# de capas, y este paso aprovecha el caché. Si `requirements.txt` no cambia,
# Docker no volverá a ejecutar la instalación de dependencias, acelerando la construcción.
COPY requirements.txt .

# Ejecuta pip para instalar todas las librerías listadas en `requirements.txt`.
# `--no-cache-dir` deshabilita el caché de pip, reduciendo el tamaño de la imagen.
# `--upgrade pip` asegura que se use la versión más reciente de pip.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- ETAPA 4: Copiar los archivos de la aplicación ---
# Copia el resto de los archivos del proyecto (app.py, modelo.pkl, etc.)
# desde el directorio local al directorio de trabajo `/app` dentro del contenedor.
COPY . .

# --- ETAPA 5: Exponer el puerto ---
# Informa a Docker que el contenedor escuchará en el puerto 5000 en tiempo de ejecución.
# Este es el puerto en el que nuestra aplicación Flask estará corriendo.
EXPOSE 5000

# --- ETAPA 6: Definir el comando de ejecución ---
# Especifica el comando que se ejecutará cuando se inicie un contenedor a partir de esta imagen.
# Se utiliza `gunicorn` que es un servidor WSGI de nivel de producción, más robusto que el
# servidor de desarrollo de Flask.
# `app:app` le dice a gunicorn que busque el objeto `app` en el archivo `app.py`.
# `--bind 0.0.0.0:5000` hace que el servidor escuche en todas las interfaces de red,
# lo cual es necesario para que el puerto mapeado por Docker funcione correctamente.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]