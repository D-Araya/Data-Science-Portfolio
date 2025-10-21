# Contenerización de una API de Machine Learning con Docker

## Objetivo del Proyecto

Este proyecto demuestra el flujo de trabajo completo para la operacionalización de un modelo de Machine Learning (MLOps), desde el entrenamiento de un modelo de clasificación hasta su despliegue como un servicio web funcional, escalable y portable, utilizando Flask para la creación de la API y Docker para la contenerización.

## Arquitectura del Repositorio

### Archivos del Proyecto

- **train_model.py**: Script de Python para entrenar un modelo `RandomForestClassifier` con el dataset "Wine" de scikit-learn y guardarlo como `modelo.pkl`.
- **modelo.pkl**: Archivo binario que contiene el modelo ya entrenado (generado por `train_model.py`).
- **app.py**: Aplicación Flask que carga `modelo.pkl` y crea una API REST con dos endpoints:
  - `/`: Muestra un mensaje de bienvenida.
  - `/predict`: Acepta peticiones POST con datos en JSON y devuelve una predicción.
- **requirements.txt**: Lista de dependencias de Python necesarias para el proyecto.
- **Dockerfile**: Receta para construir la imagen de Docker que ejecutará la aplicación.
- **.gitignore**: Archivo que especifica qué ficheros no deben ser subidos al control de versiones.

### Archivo .gitignore Recomendado

```plaintext
# Entorno Virtual
.venv/
venv/
env/

# Archivos de caché de Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Archivos de configuración de IDEs
.vscode/
.idea/

# Modelo de Machine Learning (Opcional)
# En proyectos grandes, los modelos no se suben al repo.
*.pkl
```

## Prerrequisitos

- Git instalado para clonar el repositorio
- Python 3.11.13 instalado para la configuración del entorno local
- Docker instalado y el servicio en ejecución en tu máquina

## Guía de Implementación

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/DevandMlOps/Actividad_MLOps.git
cd Actividad_MLOps
```

### Paso 2: Configuración del Entorno Virtual

#### Crear y activar el entorno virtual:

```bash
# Crear el entorno (se creará una carpeta .venv). Asegúrate de usar la versión de Python 3.11.13
# Comando para un entorno Linux
python3.11 -m venv .venv
# Comando para un entorno Windows
python -m venv .venv

# Activar el entorno en Windows (PowerShell)
.venv\Scripts\Activate.ps1

# En el símbolo del sistema (CMD)
.venv\Scripts\activate

# Activar el entorno en macOS/Linux
source .venv/bin/activate
```

#### Instalar dependencias:

```bash
pip install -r requirements.txt
```

### Paso 3: Entrenar el Modelo de Machine Learning

**Importante**: El contenedor de Docker necesita el archivo `modelo.pkl` para funcionar. Ejecuta el siguiente script para generar este archivo:

```bash
# Asegúrate de tener el entorno virtual activado
python train_model.py
```

Al finalizar, aparecerá un nuevo archivo `modelo.pkl` en la raíz del proyecto.

### Paso 4: Construir la Imagen de Docker

```bash
docker build -t ml-wine-api .
```

**Explicación del comando:**
- `docker build`: Comando para construir una imagen de Docker
- `-t ml-wine-api`: Asigna el nombre (tag) "ml-wine-api" a la imagen para identificarla fácilmente
- `.`: Especifica que el contexto de construcción es el directorio actual (donde se encuentra el Dockerfile)

Este proceso descargará la imagen base de Python e instalará las dependencias, lo cual puede tardar unos minutos la primera vez.

### Paso 5: Ejecutar el Contenedor

```bash
docker run -p 5000:5000 ml-wine-api
```

**Explicación del comando:**
- `docker run`: Comando para crear e iniciar un nuevo contenedor
- `-p 5000:5000`: Mapea el puerto 5000 de tu máquina local al puerto 5000 del contenedor
- `ml-wine-api`: Nombre de la imagen que se usará para crear el contenedor

Si todo funciona correctamente, verás los logs del servidor Gunicorn indicando que la API está en línea.

## Validación y Pruebas

### Endpoint de Bienvenida (/)

```bash
curl http://localhost:5000/
```

**Respuesta esperada:**
```json
{
  "message": "Bienvenido a la API de Clasificación de Vinos. Usa el endpoint /predict para obtener una predicción."
}
```

### Endpoint de Predicción (/predict)

```bash
curl -X POST \
-H "Content-Type: application/json" \
-d '{"features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]}' \
http://localhost:5000/predict
```

**Respuesta esperada:**
```json
{
  "prediction": "class_0"
}
```

## Solución de Problemas Comunes

### Error: "No such file or directory: modelo.pkl"
**Causa**: El archivo del modelo no existe.  
**Solución**: Ejecuta `python train_model.py` antes de construir la imagen de Docker.

### Error: "Port 5000 is already in use"
**Causa**: El puerto 5000 está siendo utilizado por otra aplicación.  
**Solución**: Usa un puerto diferente: `docker run -p 8080:5000 ml-wine-api` y accede vía `http://localhost:8080/`

### Error al activar el entorno virtual en Windows
**Causa**: Política de ejecución de PowerShell.  
**Solución**: Ejecuta `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` antes de activar el entorno.

### Error: "Docker daemon is not running"
**Causa**: El servicio de Docker no está iniciado.  
**Solución**: Inicia Docker Desktop o el servicio de Docker en tu sistema.

### Problemas con versiones de Python
**Causa**: Incompatibilidad de versiones.  
**Solución**: Verifica que tienes Python 3.11.13 instalado con `python --version`.

## Comandos Docker Útiles

```bash
# Ver imágenes construidas
docker images

# Ver contenedores en ejecución
docker ps

# Detener un contenedor
docker stop <container_id>

# Eliminar un contenedor
docker rm <container_id>

# Eliminar una imagen
docker rmi ml-wine-api
```

¡Felicidades! Has desplegado exitosamente una API de Machine Learning usando Docker siguiendo las mejores prácticas de MLOps.

---

---
[Volver al índice principal](../../README.md) | [Volver a MLOps](../README.md) | [Actividad Siguiente →](../Actividad_3_Despliegue_Automatizado/README.md)