# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import json
import joblib
import gzip

# Paso 1: Limpieza de datos
def clean_data(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.dropna()  # Eliminar registros con valores faltantes
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)
    return df

# Paso 2: División de datos en X e y
def split_data(df):
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y

# Paso 3: Crear el pipeline con MLP y PCA
def create_pipeline():
    # Columnas categóricas y numéricas
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    
    # Transformaciones
    preprocessor = ColumnTransformer(
        transformers=[ 
            ("cat", OneHotEncoder(), categorical_features),
            ("num", MinMaxScaler(), numeric_features),
        ]
    )
    
    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),  # Descomponer con PCA (todas las componentes)
            ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
            ("classifier", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)),  # MLP con 1 capa oculta de 100 neuronas
        ]
    )
    return pipeline

# Paso 4: Optimización con validación cruzada
def optimize_pipeline(X_train, y_train, pipeline):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    balanced_precision = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="balanced_accuracy"
    )
    return balanced_precision.mean()

# Paso 5: Guardar el modelo
def save_model(pipeline, path):
    with gzip.open(path, "wb") as f:
        joblib.dump(pipeline, f)

# Paso 6 y 7: Calcular métricas y matriz de confusión
def calculate_metrics_and_cm(X, y, pipeline, dataset_type):
    y_pred = pipeline.predict(X)
    metrics = {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
    }
    
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        "true_1": {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]},
    }
    return metrics, cm_dict

# Convertir todo a tipos estándar de Python antes de guardar
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj

# Ejecutar todo
def main():
    # Rutas de los archivos
    input_dir = "files/input/"
    output_dir = "files/output/"
    model_path = "files/models/model.pkl.gz"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Descomprimir y cargar datos
    train_zip_path = os.path.join(input_dir, "train_data.csv.zip")
    test_zip_path = os.path.join(input_dir, "test_data.csv.zip")
    train_inner_file = "train_default_of_credit_card_clients.csv"
    test_inner_file = "test_default_of_credit_card_clients.csv"
    
    def unzip_and_load(zip_path, inner_file):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(inner_file, input_dir)
        return pd.read_csv(os.path.join(input_dir, inner_file))
    
    train_data = unzip_and_load(train_zip_path, train_inner_file)
    test_data = unzip_and_load(test_zip_path, test_inner_file)
    
    # Limpieza
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    
    # Dividir datos
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)
    
    # Crear pipeline
    pipeline = create_pipeline()
    
    # Entrenar modelo
    pipeline.fit(X_train, y_train)
    
    # Guardar modelo
    save_model(pipeline, model_path)
    
    # Calcular métricas
    train_metrics, train_cm = calculate_metrics_and_cm(X_train, y_train, pipeline, "train")
    test_metrics, test_cm = calculate_metrics_and_cm(X_test, y_test, pipeline, "test")
    
    # Guardar métricas
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            convert_to_serializable([train_metrics, test_metrics, train_cm, test_cm]),
            f,
            indent=4,
        )

# Ejecutar script
if __name__ == "__main__":
    main()
