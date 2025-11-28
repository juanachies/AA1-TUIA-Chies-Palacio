# Deployment de Modelo de Predicción de Lluvia en Australia

Este directorio contiene los archivos necesarios para desplegar el modelo de Red Neuronal utilizando Docker.

## Contenido
- `inferencia.py`: Script que carga el modelo, preprocesa un dato de ejemplo y realiza la predicción.
- `Dockerfile`: Definición de la imagen de Docker.
- `requirements.txt`: Dependencias de Python.
- `*.joblib` y `*.keras`: Modelo entrenado y objetos de preprocesamiento (Scaler, Imputer, Columnas).

## Instrucciones

### 1. Construir la imagen (Build)
Abre una terminal en esta carpeta (`docker`) y ejecuta:

```bash
docker build -t prediccion-lluvia-tp