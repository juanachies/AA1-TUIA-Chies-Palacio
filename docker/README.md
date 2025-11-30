## Deployment de Modelo de Predicción de Lluvia en Australia

Este directorio contiene todos los archivos necesarios para construir una imagen de **Docker** y ejecutar un script de inferencia que utiliza una **Red Neuronal** pre-entrenada para predecir si lloverá, basándose en el conjunto de datos de lluvia en Australia.

---

### Estructura del Directorio y Descripción de Archivos

La carpeta `docker` contiene los siguientes archivos clave:

* **`inferencia.py`**: El script principal de inferencia. Carga el modelo (`red_neuronal.h5`), aplica el preprocesamiento a un dato de ejemplo (`ejemplo.csv`) y realiza la predicción de lluvia.
* **`Dockerfile`**: La definición de la imagen de Docker. Contiene las instrucciones para construir el entorno de ejecución, instalar dependencias y definir el comando de inicio.
* **`requirements.txt`**: Lista de dependencias de Python (bibliotecas como `pandas`, `tensorflow`, etc.) necesarias para el script.
* **`red_neuronal.h5`**: El archivo del **Modelo de Red Neuronal** pre-entrenado.
* **`scaler.joblib`**: Objeto de preprocesamiento serializado para escalar las características numéricas.
* **`mapa_regiones.joblib`**: Objeto serializado para el mapeo de variables categóricas (como las regiones).
* **`columnas.joblib`**: Archivo que almacena la lista de nombres de columnas esperadas por el modelo, asegurando el orden correcto de las características.
* **`cleaner.py`**: Script auxiliar que contiene la clase del limpiador.
* **`data_cleaner.joblib`**: Objeto de imputación o preprocesamiento complejo serializado.
* **`ejemplo.csv`**: Archivo de datos de ejemplo que se utiliza para demostrar la inferencia dentro del contenedor.
* **`README.md`**: Este archivo de documentación.

---

### Instrucciones de Despliegue con Docker

#### 1. Prerrequisitos
Asegúrate de tener **Docker Desktop** instalado y en ejecución en tu sistema. 

#### 2. Construir la Imagen (Build)

Abre una terminal en el directorio de este proyecto (`docker`) y ejecuta:

```bash
docker build -t prediccion-lluvia-tp .
```

#### 3. Ejecutar el contenedor

En la misma terminal, ejecuta:

```bash
docker run --rm prediccion-lluvia-tp
```
