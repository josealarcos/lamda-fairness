"""
Carga y preprocesado del dataset Dutch Census (Dutch Virtual Census 2001) para
experimentos de fairness con LAMDA.

Este módulo:
- carga la versión preprocesada del repositorio de Le Quy et al. (2022)
  (github.com/tailequy/fairness_dataset), usada para mantener la comparabilidad
  con las cifras de disparidad publicadas en esa fuente (SPD de una regresión
  logística sobre el atributo sex, Tabla 15 de Le Quy et al., 2022),
- limpia valores faltantes y duplicados,
- define la variable objetivo binaria a partir de `occupation`,
- construye el atributo sensible,
- codifica las variables categóricas mediante one-hot,
- devuelve X, y, sensitive y feature_names.

Objetivo
--------
La variable objetivo es `occupation`, binaria:
- y = 1 -> ocupación de alto nivel (resultado favorable, por defecto)
- y = 0 -> ocupación de bajo nivel

Atributo sensible
-----------------
`sex` (male/female). Se mantiene en X por coherencia con el resto de loaders del
proyecto.

Notas sobre las variables
-------------------------
Las columnas del fichero son códigos categóricos (posición y tamaño del hogar,
ciudadanía, país de nacimiento, nivel educativo, estatus y actividad económica,
estado civil, etc.). Se tratan como categóricas y se codifican one-hot, salvo
`age`, que se mantiene como numérica ordinal (tramo de edad). El conjunto de
columnas numéricas es configurable.

Notas sobre normalización
-------------------------
Este loader NO normaliza. La normalización min-max propia de LAMDA se realiza
dentro del clasificador, ajustada solo sobre el conjunto de entrenamiento tras
la partición train/test, evitando fuga de información desde el test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


SENSITIVE_MODE = Literal["sex", "age", "marital", "sex_age", "sex_marital"]

BASE_DIR = Path(__file__).resolve().parents[3]

# Columna objetivo del dataset Dutch preprocesado (Le Quy et al., 2022).
_TARGET_COL = "occupation"

# Columnas que se tratan como numéricas ordinales. El resto de descriptores son
# códigos categóricos y se codifican one-hot.
_NUMERIC_COLS = ["age"]

# Corte del tramo de edad empleado para construir grupos sensibles. La variable
# age es un codigo ordinal de tramo comprendido entre 4 y 15; el valor 8 reparte
# la muestra de forma practicamente equilibrada (52,8 % frente a 47,2 %). Se fija
# de antemano y sin atender a la disparidad observada.
_AGE_CUT = 8


def _get_default_path() -> Path:
    """
    Devuelve la ruta por defecto al fichero dutch.csv de Dutch Census.
    """
    return BASE_DIR / "data" / "raw" / "dutch" / "dutch.csv"


def _build_age_group(df: pd.DataFrame) -> pd.Series:
    """
    Agrupa el tramo de edad en dos categorias por el corte declarado.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Retorna
    -------
    pd.Series
        Serie de etiquetas de grupo de edad.
    """
    edad = pd.to_numeric(df["age"], errors="coerce")
    etiquetas = np.where(edad <= _AGE_CUT, f"age<={_AGE_CUT}", f"age>{_AGE_CUT}")
    return pd.Series(etiquetas, index=df.index)


def _build_marital_group(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve el estado civil como categoria, sin recodificar sus valores.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Retorna
    -------
    pd.Series
        Serie de etiquetas de estado civil.
    """
    return "marital" + df["marital_status"].astype(str)


def _build_sensitive(df: pd.DataFrame, mode: SENSITIVE_MODE) -> np.ndarray:
    """
    Construye el atributo sensible segun el modo elegido.

    El conjunto contiene, ademas del sexo, otras variables susceptibles de
    actuar como atributo sensible. Se exponen la edad, agrupada en dos tramos
    por el corte declarado, y el estado civil con sus cuatro categorias
    originales, asi como los dos modos compuestos que resultan de cruzarlas con
    el sexo. El modo por defecto sigue siendo el sexo, que es el que emplea la
    fuente de referencia y sobre el que se construye el panel principal.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    mode : {"sex", "age", "marital", "sex_age", "sex_marital"}
        Modo de construccion del atributo sensible.

    Retorna
    -------
    np.ndarray
        Vector del atributo sensible.
    """
    sexo = df["sex"].astype(str)

    if mode == "sex":
        return sexo.to_numpy()

    if mode == "age":
        return _build_age_group(df).to_numpy()

    if mode == "marital":
        return _build_marital_group(df).to_numpy()

    if mode == "sex_age":
        return (sexo + "_" + _build_age_group(df)).to_numpy()

    if mode == "sex_marital":
        return (sexo + "_" + _build_marital_group(df)).to_numpy()

    raise ValueError(
        "Modo sensible no valido para Dutch. Validos: 'sex', 'age', 'marital', "
        "'sex_age' y 'sex_marital'."
    )


def load_dutch_dataset(
    file_path: str | Path | None = None,
    sensitive_mode: SENSITIVE_MODE = "sex",
    favorable_high: bool = True,
    numeric_cols: list[str] | None = None,
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carga y preprocesa el dataset Dutch Census a partir de dutch.csv.

    Parámetros
    ----------
    file_path : str | Path | None, default=None
        Ruta al fichero dutch.csv. Si es None, se usa la ruta por defecto
        (data/raw/dutch/dutch.csv).
    sensitive_mode : {"sex", "age", "marital", "sex_age", "sex_marital"}, default="sex"
        Modo de construcción del atributo sensible.
    favorable_high : bool, default=True
        Si True, y = 1 corresponde a ocupación de alto nivel (favorable).
        Si False, se invierte la codificación del objetivo.
    numeric_cols : list[str] | None, default=None
        Columnas a tratar como numéricas ordinales. Si None, se usa
        _NUMERIC_COLS. El resto de descriptores se codifican one-hot.
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
        raise FileNotFoundError(
            f"No se encontró el dataset en: {file_path}. "
            "Descárgalo de github.com/tailequy/fairness_dataset "
            "(experiments/data/dutch.csv) y colócalo en data/raw/dutch/."
        )

    df = pd.read_csv(file_path)

    if drop_duplicates:
        df = df.drop_duplicates()

    df = df[df[_TARGET_COL].notna()]

    if drop_na:
        df = df.dropna()

    if df.empty:
        raise ValueError("Dataset vacío tras limpieza.")

    y_raw = df[_TARGET_COL].astype(int).to_numpy()
    y = y_raw if favorable_high else (1 - y_raw)

    sensitive = _build_sensitive(df, sensitive_mode)

    feature_cols = [c for c in df.columns if c != _TARGET_COL]
    X_df = df[feature_cols].copy()

    num_cols = list(numeric_cols) if numeric_cols is not None else list(_NUMERIC_COLS)
    num_cols = [c for c in num_cols if c in X_df.columns]
    categorical_cols = [c for c in feature_cols if c not in num_cols]

    # Numéricas ordinales (sin normalizar; lo hace el clasificador).
    if num_cols:
        X_num = X_df[num_cols].to_numpy(dtype=float)
    else:
        X_num = np.empty((len(X_df), 0), dtype=float)

    # Categóricas codificadas one-hot (incluye sex, que se mantiene en X).
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X_df[categorical_cols].astype(str))
        cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        X_cat = np.empty((len(X_df), 0), dtype=float)
        cat_feature_names = []

    feature_names = num_cols + cat_feature_names
    X = np.hstack([X_num, X_cat]).astype(float)

    return X, y, sensitive, feature_names


if __name__ == "__main__":
    X, y, s, features = load_dutch_dataset()

    print("✔ Dataset Dutch Census cargado correctamente")
    print("Shape X:", X.shape)
    print("Nº features:", len(features))
    print("Clases en y:", np.unique(y, return_counts=True))
    print("Grupos sensibles:", np.unique(s, return_counts=True))
