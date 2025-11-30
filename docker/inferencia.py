import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

from cleaner import DataCleaner


def cargar_artefactos():     
    model = tf.keras.models.load_model('red_neuronal.h5')
    scaler = joblib.load('scaler.joblib')
    cleaner = joblib.load('data_cleaner.joblib')
    model_columns = joblib.load('columnas.joblib')
    mapa_regiones = joblib.load('mapa_regiones.joblib')
    
    return model, scaler, cleaner, model_columns, mapa_regiones


def preprocesar_datos(df, scaler, cleaner, model_columns, mapa_regiones):
    df = df.copy()

    # Convertir date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Creacion de region
    if 'Location' in df.columns:
        df['Region'] = df['Location'].map(mapa_regiones).fillna(0)

    # Imputacion
    df_clear = cleaner.transform(df)

    # Codificación
    df_processed = pd.get_dummies(df_clear, drop_first=True)

    # Alinear columnas del modelo
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)

    # Escalado
    data_scaled = scaler.transform(df_processed)

    return data_scaled


def predict(df):
    model, scaler, cleaner, model_columns, mapa_regiones = cargar_artefactos()

    X = preprocesar_datos(df, scaler, cleaner, model_columns, mapa_regiones)

    preds = model.predict(X, verbose=0)

    # Probabilidad de lluvia
    p_lluvia = preds[:, 1]

    # Predicción final
    y_pred = (p_lluvia >= 0.5).astype(int)

    resultado = "llueve" if y_pred == 1 else "no llueve"

    print(f"Probabilidad de lluvia: {p_lluvia[0]*100:.2f}% | Predicción: {resultado} mañana ")

    return y_pred, p_lluvia


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