"""
Baselines de ML estándar (sin mitigación) para la fase experimental.

Este módulo entrena clasificadores estándar de scikit-learn sobre el mismo split
y preprocesado que LAMDA, y calcula sus métricas de rendimiento y de equidad con
las MISMAS funciones de reporte del proyecto (reporting.summarize_model_metrics).
Su función es situar a LAMDA respecto a modelos habituales, según la ampliación
de alcance solicitada por el tutor (punto b: otras técnicas de ML sobre el
dataset directo, sin mitigar).

Encuadre metodológico
---------------------
Estos modelos NO son técnicas de fairness. Se incluyen como REFERENCIA de
rendimiento y disparidad de clasificadores estándar sin mitigar. Sirven además
para diagnosticar la aprendibilidad de cada dataset: si un modelo estándar supera
el baseline de clase mayoritaria y LAMDA no, el problema es de LAMDA
(configuración) y no del dataset.

Separación de responsabilidades
-------------------------------
Este módulo NO contiene lógica de mitigación ni de LAMDA. Solo reutiliza la
carga/partición (runner.load_and_split), el perfil de métricas
(metric_profiles.build_metric_profile) y el reporte (reporting), de modo que las
filas de baselines son directamente comparables con las de detección y
mitigación (mismas columnas).

Anti-fuga
---------
La normalización min-max se ajusta SOLO sobre el conjunto de entrenamiento
(coherente con la normalización interna de LAMDA), y se aplica congelada a test.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from lamda.experiments import runner
from lamda.experiments import config as cfg
from lamda.experiments import reporting
from lamda.fairness.disparity import FairnessThresholds
from lamda.fairness.metric_profiles import build_metric_profile


# =========================================================
# ESTIMADORES ESTÁNDAR
# =========================================================

def build_estimators(random_state: int = cfg.RANDOM_STATE) -> dict:
    """
    Construye el diccionario de clasificadores estándar de referencia.

    Se emplean tres modelos habituales en la literatura de referencia sobre
    estos datasets: regresión logística (usada por Le Quy et al., 2022, para
    calcular las métricas de fairness publicadas), árbol de decisión y random
    forest. Todos con la misma semilla para reproducibilidad.

    Parámetros
    ----------
    random_state : int
        Semilla común de los estimadores.

    Retorna
    -------
    dict
        Diccionario nombre -> estimador sin ajustar.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }


# =========================================================
# BASELINE TRIVIAL (CLASE MAYORITARIA)
# =========================================================

def _majority_row(
    y_train,
    y_test,
    s_test,
    task: str,
    thresholds: FairnessThresholds,
    weights,
    class_aggregation: str,
) -> dict:
    """
    Construye la fila del predictor de clase mayoritaria (referencia de
    aprendibilidad). Un modelo que no supere esta fila en accuracy no está
    aprendiendo. Un predictor constante es, además, trivialmente "equitativo"
    (asigna la misma clase a todos los grupos), lo que ilustra por qué una baja
    disparidad no implica un buen modelo.
    """
    majority_class = Counter(np.asarray(y_train).tolist()).most_common(1)[0][0]
    y_pred = np.full(shape=np.asarray(y_test).shape, fill_value=majority_class,
                     dtype=np.asarray(y_test).dtype)

    row = reporting.summarize_model_metrics(
        y_test, y_pred, s_test, task=task,
        thresholds=thresholds, weights=weights,
        class_aggregation=class_aggregation,
    )
    return {"Modelo": "Clase mayoritaria", **row}


# =========================================================
# EJECUCIÓN DE BASELINES PARA UN DATASET
# =========================================================

def run_baselines(
    dataset_name: str,
    sensitive_mode: str | None = None,
    thresholds: FairnessThresholds | None = None,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
    file_path: str | None = None,
    metric_profile_kind: str | None = None,
    include_majority: bool = True,
    drop_sensitive_from_X: bool = False,
) -> dict:
    """
    Entrena los baselines estándar sobre un dataset y devuelve su tabla de
    métricas, comparable con las de detección y mitigación.

    Usa exactamente la misma partición que LAMDA (runner.load_and_split con el
    mismo random_state), de modo que la comparación es sobre el mismo test.

    Parámetros
    ----------
    dataset_name : str
        Identificador del dataset (ver config.DATASETS).
    sensitive_mode : str | None
        Atributo sensible. Si None, el de config.
    thresholds : FairnessThresholds | None
        Umbrales de equidad. Si None, los de config.
    test_size, random_state : partición (deben coincidir con los de LAMDA).
    file_path : str | None
        Ruta del fichero de datos, si aplica.
    metric_profile_kind : str | None
        Fuerza el perfil ("binary"/"multiclass"). Si None, se deriva del nº de
        clases.
    include_majority : bool
        Si True, añade la fila del predictor de clase mayoritaria.

    Retorna
    -------
    dict
        {"table": DataFrame, "predictions": dict, "majority_accuracy": float,
         "task": str, "metadata": dict}
    """
    (X_train, X_test, y_train, y_test,
     s_train, s_test, feature_names, dataset_config) = runner.load_and_split(
        dataset_name=dataset_name,
        sensitive_mode=sensitive_mode,
        test_size=test_size,
        random_state=random_state,
        file_path=file_path,
        drop_sensitive_from_X=drop_sensitive_from_X,
    )

    n_classes = len(np.unique(y_train))
    profile = build_metric_profile(n_classes=n_classes, kind=metric_profile_kind)
    task = profile.kind
    weights = profile.weights
    class_aggregation = profile.class_aggregation

    if thresholds is None:
        thresholds = cfg.FAIRNESS_THRESHOLDS

    # Normalización min-max ajustada SOLO sobre train (coherente con LAMDA).
    scaler = MinMaxScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    estimators = build_estimators(random_state=random_state)

    rows = []
    predictions = {"y_test": y_test, "sensitive_test": s_test}

    for name, est in estimators.items():
        est.fit(X_train_s, y_train)
        y_pred = est.predict(X_test_s)
        predictions[name] = y_pred

        r = reporting.summarize_model_metrics(
            y_test, y_pred, s_test, task=task,
            thresholds=thresholds, weights=weights,
            class_aggregation=class_aggregation,
        )
        rows.append({"Modelo": name, **r})

    # Referencia de aprendibilidad: clase mayoritaria.
    majority_class = Counter(np.asarray(y_train).tolist()).most_common(1)[0][0]
    majority_accuracy = float(
        (np.full(np.asarray(y_test).shape, majority_class,
                 dtype=np.asarray(y_test).dtype) == np.asarray(y_test)).mean()
    )

    if include_majority:
        rows.append(_majority_row(
            y_train, y_test, s_test, task=task,
            thresholds=thresholds, weights=weights,
            class_aggregation=class_aggregation,
        ))

    baselines_table = pd.DataFrame(rows).set_index("Modelo")

    metadata = {
        "dataset": dataset_name,
        "fase": "baselines",
        "sensitive_mode": sensitive_mode or dataset_config.sensitive_mode,
        "task": task,
        "n_classes": int(n_classes),
        "modelos": list(estimators.keys()),
        "majority_accuracy": majority_accuracy,
        "drop_sensitive_from_X": drop_sensitive_from_X,
        "n_features": int(X_train.shape[1]),
        "test_size": test_size,
        "random_state": random_state,
    }

    return {
        "table": baselines_table,
        "predictions": predictions,
        "majority_accuracy": majority_accuracy,
        "task": task,
        "metadata": metadata,
    }


# =========================================================
# PERSISTENCIA
# =========================================================

def persist_baselines(
    dataset_name: str,
    result: dict,
    results_dir: str = cfg.RESULTS_DIR,
) -> str:
    """
    Persiste la tabla de baselines en {dataset}_baselines.csv, en el mismo
    formato que las tablas de detección y mitigación, para que el notebook de
    agregación pueda leerla con load_csv(name, "baselines").

    Retorna
    -------
    str
        Ruta del fichero escrito.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{dataset_name}_baselines.csv")
    result["table"].to_csv(path, index=True)
    return path
