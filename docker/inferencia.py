import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Desactivar logs de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cargar_artefactos():
    model_path = 'red_neuronal.keras'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no se encuentra en: {os.path.abspath(model_path)}")
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load('scaler.joblib')
    cleaner = joblib.load('data_cleaner.joblib')
    model_columns = joblib.load('columnas.joblib')
    mapa_regiones = joblib.load('mapa_regiones.joblib')
    
    return model, scaler, cleaner, model_columns, mapa_regiones


def preprocesar_datos(df, scaler, cleaner, model_columns, mapa_regiones):
    # Agregado de Region
    if 'Location' in df.columns:
        df['Region'] = df['Location'].map(mapa_regiones).fillna(0)
    
    # Codificación de variables categóricas
    if 'RainToday' in df.columns:
        df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})

    drop = ['Location']
    df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')

    df_processed = pd.get_dummies(df, drop_first=True)

    # Alinear columnas con las del entrenamiento
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)

    # Imputación
    data_clear = cleaner.transform(df_processed)

    data_scaled = scaler.transform(data_clear)
    
    return data_scaled


def predict(df):
    model, scaler, cleaner, model_columns, mapa_regiones = cargar_artefactos()

    df_procesado = preprocesar_datos(df, scaler, cleaner, model_columns, mapa_regiones)

    predicciones = model.predict(df_procesado, verbose=0)
    y_pred = np.argmax(predicciones, axis=1)

    for i, clase in enumerate(y_pred):
        probabilidad = predicciones[i][clase] * 100
        resultado = "LLUEVE" if clase == 1 else "NO LLUEVE"
        print(f"Fila {i} -> Predicción: {resultado} mañana | Probabilidad: {probabilidad:.2f}%")

    # y_pred, y_proba = model.predict(df_procesado)

    # resultado = "LLUEVE" if predicciones == 1 else "NO LLUEVE"
    # probabilidad = y_proba * 100

    # print(f"Predicción: {resultado} mañana")
    # print(f"Probabilidad: {probabilidad:.2f}%")

    return y_pred, predicciones


def main(ejemplo):
    try:
        df = pd.read_csv(ejemplo)
        pred, proba = predict(df)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main('ejemplo.csv')