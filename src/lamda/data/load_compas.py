"""
Carga y preprocesado del dataset COMPAS para experimentos de fairness con LAMDA.

Este módulo:
- carga el dataset COMPAS
- aplica los filtros habituales del benchmark
- limpia valores faltantes
- define la variable objetivo
- construye el atributo sensible
- codifica variables categóricas (One-Hot)
- devuelve X, y, sensitive y feature_names
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


SENSITIVE_MODE = Literal["sex", "race", "sex_race"]

BASE_DIR = Path(__file__).resolve().parents[3]


def _get_default_path() -> Path:
    """
    Devuelve la ruta por defecto al dataset COMPAS.
    """
    return BASE_DIR / "data" / "raw" / "compas" / "compas-scores-two-years.csv"


def _build_sensitive(df: pd.DataFrame, mode: SENSITIVE_MODE) -> np.ndarray:
    """
    Construye el atributo sensible según el modo elegido.

    En los modos que incluyen race, se asume que el DataFrame ha sido
    previamente restringido a las dos razas mayoritarias
    {African-American, Caucasian}, igual que en el preprocesado experimental
    de Le Quy et al. (2022) sobre COMPAS. El filtrado se realiza en la carga
    cuando binarize_race=True.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    mode : {"sex", "race", "sex_race"}
        Modo de construcción del atributo sensible.

    Retorna
    -------
    np.ndarray
        Vector del atributo sensible.
    """
    if mode == "sex":
        sensitive = df["sex"].astype(str)

    elif mode == "race":
        sensitive = df["race"].astype(str)

    elif mode == "sex_race":
        sensitive = df["sex"].astype(str) + "_" + df["race"].astype(str)

    else:
        raise ValueError("Modo sensible no válido.")

    return sensitive.to_numpy()


def _apply_compas_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los filtros habituales del benchmark COMPAS.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.

    Retorna
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    df = df.copy()

    df = df[df["days_b_screening_arrest"].between(-30, 30, inclusive="both")]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]

    return df


def _compas_feature_columns() -> list[str]:
    """
    Devuelve las columnas seleccionadas como features en COMPAS.

    Retorna
    -------
    list[str]
        Lista de columnas de entrada.
    """
    return [
        "age",
        "age_cat",
        "sex",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
    ]


def load_compas_dataset(
    file_path: str | Path | None = None,
    sensitive_mode: SENSITIVE_MODE = "race",
    drop_na: bool = True,
    drop_duplicates: bool = True,
    apply_filters: bool = True,
    binarize_race: bool = True,
    favorable_no_recid: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carga y preprocesa el dataset COMPAS.

    Parámetros
    ----------
    file_path : str | Path | None, default=None
        Ruta al fichero. Si es None, se usa la ruta por defecto.
    sensitive_mode : {"sex", "race", "sex_race"}, default="race"
        Modo de construcción del atributo sensible. Por defecto "race", que
        es el atributo usado por Le Quy et al. (2022) para COMPAS.
    drop_na : bool, default=True
        Si True, elimina filas con valores faltantes.
    drop_duplicates : bool, default=True
        Si True, elimina filas duplicadas.
    apply_filters : bool, default=True
        Si True, aplica los filtros estándar del benchmark ProPublica.
    binarize_race : bool, default=True
        Si True, restringe el dataset a las dos razas mayoritarias
        {African-American, Caucasian}, replicando el preprocesado
        experimental de Le Quy et al. (2022), que filtra el conjunto a esos
        dos grupos antes de entrenar. Descarta 894 observaciones sobre las
        6.172 que resultan de los filtros de ProPublica. AIF360 dicotomiza de
        otro modo, reetiquetando a {Caucasian, Not Caucasian} sin descartar
        filas; aquí se sigue a Le Quy por ser la fuente con la que se
        contrasta la disparidad de partida.
    favorable_no_recid : bool, default=True
        Define cuál es el resultado favorable de cara a las métricas de
        equidad. Si True (recomendado), el resultado favorable es NO reincidir,
        de modo que y=1 corresponde a two_year_recid=0. Esta es la convención
        de AIF360 y de la mayoría de trabajos aplicados: el resultado deseable
        para el individuo es no ser predicho como reincidente. Si False, se
        mantiene la codificación original (y=1 = reincide), habitual en algunos
        trabajos metodológicos. La elección afecta al signo de SPD y EOD, por
        lo que en la fase de detección conviene comparar la magnitud (|SPD|)
        con la literatura.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
        X, y, sensitive y feature_names.

    Notas sobre normalización
    -------------------------
    Este loader NO normaliza las variables numéricas; la normalización min-max
    de LAMDA se realiza en el clasificador, ajustada solo sobre entrenamiento.
    """
    if file_path is None:
        file_path = _get_default_path()
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

    df = pd.read_csv(file_path)

    if apply_filters:
        df = _apply_compas_filters(df)

    if binarize_race:
        # Se restringe a las dos razas mayoritarias para reproducir el
        # preprocesado con el que se contrasta la disparidad de partida.
        df = df[df["race"].isin(["African-American", "Caucasian"])]

    if drop_na:
        df = df.dropna(subset=_compas_feature_columns() + ["two_year_recid"])

    if drop_duplicates:
        df = df.drop_duplicates()

    if df.empty:
        raise ValueError("Dataset vacío tras limpieza.")

    recid = df["two_year_recid"].astype(int)

    # El resultado favorable para el individuo es no reincidir. Por defecto se
    # codifica y=1 = "no reincide" (favorable), invirtiendo two_year_recid,
    # para alinear la clase favorable con la convención de AIF360 y facilitar
    # la comparación con la literatura. Con favorable_no_recid=False se
    # mantiene la codificación original (y=1 = reincide).
    if favorable_no_recid:
        y = (1 - recid).to_numpy()
    else:
        y = recid.to_numpy()

    sensitive = _build_sensitive(df, sensitive_mode)

    X_df = df[_compas_feature_columns()].copy()

    categorical_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_df.select_dtypes(exclude=["object"]).columns.tolist()

    X_num = X_df[numerical_cols].to_numpy(dtype=float)

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    X_cat = encoder.fit_transform(X_df[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()

    feature_names = numerical_cols + cat_feature_names
    X = np.hstack([X_num, X_cat]).astype(float)

    return X, y, sensitive, feature_names


if __name__ == "__main__":
    X, y, s, features = load_compas_dataset()

    print("✔ Dataset COMPAS cargado correctamente")
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Shape sensitive:", s.shape)
    print("Nº features:", len(features))
    print("Clases en y:", np.unique(y))
    print("Grupos sensibles:", np.unique(s))