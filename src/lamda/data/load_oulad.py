"""
Carga y preprocesado del dataset OULAD (Open University Learning Analytics Dataset)
para experimentos de fairness con LAMDA.

Este módulo:
- carga el fichero studentInfo.csv de OULAD (Kuzilek et al., 2017)
- limpia valores faltantes y duplicados
- define la variable objetivo a partir de `final_result`
- construye el atributo sensible (individual o interseccional)
- codifica las variables categóricas mediante one-hot y normaliza las numéricas
- devuelve X, y, sensitive y feature_names

Notas sobre el objetivo
-----------------------
La variable objetivo original `final_result` es categórica con cuatro niveles:

- Distinction
- Pass
- Fail
- Withdrawn

Por defecto se trata como un problema multiclase de cuatro clases, codificadas
como enteros mediante un mapeo ordinal de menor a mayor éxito académico:

- y = 0 -> Withdrawn
- y = 1 -> Fail
- y = 2 -> Pass
- y = 3 -> Distinction

Este es el caso de mayor interés, ya que constituye una tarea multiclase genuina
sobre la que evaluar la descomposición one-vs-rest de las métricas de fairness.
También se ofrece una opción binaria (pass/fail) que agrupa {Pass, Distinction}
frente a {Fail, Withdrawn}, habitual en la literatura.

Notas sobre las variables
-------------------------
Se utiliza el fichero studentInfo.csv, que contiene los datos demográficos y el
resultado final de cada estudiante. La columna identificativa `id_student` se
descarta. Cuando un estudiante aparece en varias presentaciones de módulo, las
filas se mantienen como observaciones independientes, salvo que se eliminen
duplicados exactos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


SENSITIVE_MODE = Literal[
    "gender",
    "disability",
    "age_band",
    "gender_disability",
    "gender_age_band",
]
TARGET_MODE = Literal["multiclass", "binary"]
BINARY_POSITIVE_CLASS = Literal["pass", "fail"]

BASE_DIR = Path(__file__).resolve().parents[3]


def _get_default_path() -> Path:
    """
    Devuelve la ruta por defecto al fichero studentInfo.csv de OULAD.
    """
    return BASE_DIR / "data" / "raw" / "oulad" / "studentInfo.csv"


def _target_column() -> str:
    """
    Devuelve el nombre de la variable objetivo original.
    """
    return "final_result"


def _id_columns() -> list[str]:
    """
    Devuelve las columnas identificativas que no se usan como variables predictoras.

    `id_student` identifica al estudiante y no debe utilizarse como descriptor.
    """
    return ["id_student"]


def _multiclass_target_mapping() -> dict[str, int]:
    """
    Devuelve el mapeo ordinal de la variable objetivo multiclase.

    Se ordena de menor a mayor éxito académico.
    """
    return {
        "Withdrawn": 0,
        "Fail": 1,
        "Pass": 2,
        "Distinction": 3,
    }


def _build_sensitive(df: pd.DataFrame, mode: SENSITIVE_MODE) -> np.ndarray:
    """
    Construye el atributo sensible según el modo elegido.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    mode : {"gender", "disability", "age_band", "gender_disability", "gender_age_band"}
        Modo de construcción del atributo sensible.
        - "gender"            : género del estudiante (M / F).
        - "disability"        : indicador de discapacidad (Y / N).
        - "age_band"          : tramo de edad (0-35, 35-55, 55<=).
        - "gender_disability" : combinación interseccional género x discapacidad.
        - "gender_age_band"   : combinación interseccional género x tramo de edad.

    Retorna
    -------
    np.ndarray
        Vector del atributo sensible.
    """
    gender_group = df["gender"].astype(str)
    disability_group = df["disability"].astype(str)
    age_group = df["age_band"].astype(str)

    if mode == "gender":
        sensitive = gender_group

    elif mode == "disability":
        sensitive = disability_group

    elif mode == "age_band":
        sensitive = age_group

    elif mode == "gender_disability":
        sensitive = gender_group + "_" + disability_group

    elif mode == "gender_age_band":
        sensitive = gender_group + "_" + age_group

    else:
        raise ValueError("Modo sensible no válido.")

    return sensitive.to_numpy()


def _oulad_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas seleccionadas como features en OULAD.

    Se eliminan la columna identificativa y la variable objetivo. Los atributos
    sensibles (gender, disability, age_band) se mantienen en X por coherencia
    con el resto de loaders del proyecto.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Retorna
    -------
    list[str]
        Lista de nombres de columnas usadas como features.
    """
    excluded_cols = set(_id_columns() + [_target_column()])

    return [col for col in df.columns if col not in excluded_cols]


def _build_multiclass_target(y_raw: pd.Series) -> np.ndarray:
    """
    Construye la variable objetivo multiclase a partir de `final_result`.

    Aplica el mapeo ordinal {Withdrawn, Fail, Pass, Distinction} -> {0, 1, 2, 3}.

    Parámetros
    ----------
    y_raw : pd.Series
        Variable objetivo categórica original.

    Retorna
    -------
    np.ndarray
        Vector objetivo entero.
    """
    mapping = _multiclass_target_mapping()
    y_clean = y_raw.astype(str).str.strip()

    unknown = set(y_clean.unique()) - set(mapping.keys())
    if unknown:
        raise ValueError(f"Valores de final_result no reconocidos: {unknown}")

    return y_clean.map(mapping).astype(int).to_numpy()


def _build_binary_target(
    y_raw: pd.Series,
    positive_class: BINARY_POSITIVE_CLASS = "pass",
) -> np.ndarray:
    """
    Construye una variable objetivo binaria a partir de `final_result`.

    Agrupa {Pass, Distinction} como éxito y {Fail, Withdrawn} como no éxito.

    Parámetros
    ----------
    y_raw : pd.Series
        Variable objetivo categórica original.
    positive_class : {"pass", "fail"}, default="pass"
        Clase que se codifica como 1.
        - "pass" : 1 = {Pass, Distinction}.
        - "fail" : 1 = {Fail, Withdrawn}.

    Retorna
    -------
    np.ndarray
        Vector objetivo binario.
    """
    y_clean = y_raw.astype(str).str.strip()
    success = y_clean.isin(["Pass", "Distinction"])

    if positive_class == "pass":
        return success.astype(int).to_numpy()

    if positive_class == "fail":
        return (~success).astype(int).to_numpy()

    raise ValueError("Clase positiva binaria no válida.")


def load_oulad_dataset(
    file_path: str | Path | None = None,
    sensitive_mode: SENSITIVE_MODE = "gender",
    target_mode: TARGET_MODE = "multiclass",
    binary_positive_class: BINARY_POSITIVE_CLASS = "pass",
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carga y preprocesa el dataset OULAD a partir de studentInfo.csv.

    Parámetros
    ----------
    file_path : str | Path | None, default=None
        Ruta al fichero studentInfo.csv. Si es None, se usa la ruta por defecto.
    sensitive_mode : {"gender", "disability", "age_band", "gender_disability", "gender_age_band"}, default="gender"
        Modo de construcción del atributo sensible.
    target_mode : {"multiclass", "binary"}, default="multiclass"
        Modo de construcción de la variable objetivo.
        - "multiclass" : cuatro clases (Withdrawn, Fail, Pass, Distinction).
        - "binary"     : éxito {Pass, Distinction} frente a {Fail, Withdrawn}.
    binary_positive_class : {"pass", "fail"}, default="pass"
        Clase positiva si target_mode="binary".
    drop_na : bool, default=True
        Si True, elimina filas con valores faltantes.
    drop_duplicates : bool, default=True
        Si True, elimina filas duplicadas.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
        X, y, sensitive y feature_names.

    Notas sobre normalización
    -------------------------
    Este loader NO normaliza las variables numéricas. La normalización
    min-max propia de LAMDA se realiza dentro del clasificador
    (LamdaMinMaxNormalizer), que se ajusta únicamente sobre el conjunto de
    entrenamiento tras la partición train/test. Normalizar aquí, sobre el
    dataset completo antes de la partición, introduciría fuga de información
    del conjunto de test en el preprocesado. Las variables numéricas se
    devuelven en su escala original y las categóricas codificadas one-hot en
    {0, 1}; el clasificador se encarga de la normalización posterior.
    """
    if file_path is None:
        file_path = _get_default_path()
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

    df = pd.read_csv(file_path)

    if drop_duplicates:
        df = df.drop_duplicates()

    target_col = _target_column()

    # Se eliminan filas sin objetivo válido.
    df = df[df[target_col].notna()]

    if drop_na:
        df = df.dropna()

    if df.empty:
        raise ValueError("Dataset vacío tras limpieza.")

    y_raw = df[target_col]

    if target_mode == "multiclass":
        y = _build_multiclass_target(y_raw)

    elif target_mode == "binary":
        y = _build_binary_target(
            y_raw=y_raw,
            positive_class=binary_positive_class,
        )

    else:
        raise ValueError("Modo de objetivo no válido.")

    sensitive = _build_sensitive(df, sensitive_mode)

    feature_cols = _oulad_feature_columns(df)
    X_df = df[feature_cols].copy()

    # Separación de variables categóricas y numéricas.
    categorical_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_df.select_dtypes(exclude=["object"]).columns.tolist()

    # Las variables numéricas se devuelven en su escala original, sin
    # normalizar. La normalización min-max de LAMDA se aplica en el
    # clasificador, ajustada solo sobre entrenamiento, para no introducir
    # fuga de información desde el test.
    if len(numerical_cols) > 0:
        X_num = X_df[numerical_cols].to_numpy(dtype=float)
    else:
        X_num = np.empty((len(X_df), 0), dtype=float)

    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        X_cat = encoder.fit_transform(X_df[categorical_cols])
        cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        X_cat = np.empty((len(X_df), 0), dtype=float)
        cat_feature_names = []

    feature_names = numerical_cols + cat_feature_names
    X = np.hstack([X_num, X_cat]).astype(float)

    return X, y, sensitive, feature_names


if __name__ == "__main__":
    X, y, s, features = load_oulad_dataset()

    print("✔ Dataset OULAD cargado correctamente")
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Shape sensitive:", s.shape)
    print("Nº features:", len(features))
    print("Clases en y:", np.unique(y, return_counts=True))
    print("Grupos sensibles:", np.unique(s, return_counts=True))
