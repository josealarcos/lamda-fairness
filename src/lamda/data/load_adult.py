"""
Carga y preprocesado del dataset Adult para experimentos de fairness con LAMDA.

Este módulo:
- carga el dataset Adult
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
    Devuelve la ruta por defecto al dataset Adult.
    """
    return BASE_DIR / "data" / "raw" / "adult" / "adult.data"


def _adult_column_names() -> list[str]:
    """
    Devuelve los nombres de columnas del dataset Adult.
    """
    return [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]


def _binarize_race(race: pd.Series) -> pd.Series:
    """
    Binariza la variable race en {White, NonWhite}.

    La literatura de fairness sobre Adult (Le Quy et al., 2022, siguiendo a
    Zafar et al. y otros) codifica race como binaria {white, non-white}, en
    lugar de emplear las cinco categorías originales. Se adopta esta
    binarización para que el atributo sensible tenga dos grupos, como en la
    literatura de referencia, y la disparidad sea comparable en orden de
    magnitud.

    Parámetros
    ----------
    race : pd.Series
        Serie original de raza (cinco categorías).

    Retorna
    -------
    pd.Series
        Serie binaria {"White", "NonWhite"}.
    """
    values = race.astype(str).str.strip()
    return pd.Series(
        np.where(values == "White", "White", "NonWhite"),
        index=race.index,
    )


def _build_sensitive(df: pd.DataFrame, mode: SENSITIVE_MODE) -> np.ndarray:
    """
    Construye el atributo sensible según el modo elegido.

    En los modos que incluyen race, esta se binariza en {White, NonWhite}
    siguiendo la convención de la literatura de fairness sobre Adult, de modo
    que la disparidad sea comparable con los trabajos de referencia.

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
        sensitive = _binarize_race(df["race"])

    elif mode == "sex_race":
        sensitive = df["sex"].astype(str) + "_" + _binarize_race(df["race"])

    else:
        raise ValueError("Modo sensible no válido.")

    return sensitive.to_numpy()


def _adult_feature_columns() -> list[str]:
    """
    Devuelve las columnas seleccionadas como features en Adult.

    Se excluyen dos columnas respecto al conjunto original:

    - `fnlwgt` (final weight): es un peso muestral asociado al diseño de la
      encuesta, no una característica del individuo. La literatura de fairness
      sobre Adult lo descarta sistemáticamente (Le Quy et al., 2022; Kamiran &
      Calders, 2012).
    - `education`: es la versión categórica de `education_num`, que codifica el
      mismo nivel educativo de forma numérica ordinal. Se mantiene únicamente
      `education_num` para evitar duplicar la misma información y no inflar
      innecesariamente la dimensionalidad tras la codificación one-hot.

    Retorna
    -------
    list[str]
        Lista de columnas de entrada.
    """
    return [
        "age",
        "workclass",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
    ]


def load_adult_dataset(
    file_path: str | Path | None = None,
    sensitive_mode: SENSITIVE_MODE = "sex",
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carga y preprocesa el dataset Adult.

    Parámetros
    ----------
    file_path : str | Path | None, default=None
        Ruta al fichero. Si es None, se usa la ruta por defecto.
    sensitive_mode : {"sex", "race", "sex_race"}, default="sex"
        Modo de construcción del atributo sensible. Por defecto "sex", que es
        el atributo único usado por Le Quy et al. (2022) para Adult y permite
        comparar la disparidad de partida con la literatura. En los modos con
        race, esta se binariza en {White, NonWhite}. El modo interseccional
        "sex_race" queda disponible para análisis adicionales.
    drop_na : bool, default=True
        Si True, elimina filas con valores faltantes.
    drop_duplicates : bool, default=True
        Si True, elimina filas duplicadas.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
        X, y, sensitive y feature_names.
    """
    if file_path is None:
        file_path = _get_default_path()
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

    df = pd.read_csv(
        file_path,
        header=None,
        names=_adult_column_names(),
        na_values="?",
        skipinitialspace=True,
    )

    if drop_na:
        df = df.dropna()

    if drop_duplicates:
        df = df.drop_duplicates()

    if df.empty:
        raise ValueError("Dataset vacío tras limpieza.")

    income_clean = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    y = (income_clean == ">50K").astype(int).to_numpy()

    sensitive = _build_sensitive(df, sensitive_mode)

    X_df = df[_adult_feature_columns()].copy()

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
    X, y, s, features = load_adult_dataset()

    print("✔ Dataset Adult cargado correctamente")
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Shape sensitive:", s.shape)
    print("Nº features:", len(features))
    print("Clases en y:", np.unique(y))
    print("Grupos sensibles:", np.unique(s))