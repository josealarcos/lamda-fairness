"""
Carga y preprocesado del dataset Student Performance para experimentos de fairness con LAMDA.

Este módulo:
- carga el dataset Student Performance (Cortez & Silva, 2008; UCI id 320)
- limpia valores faltantes y duplicados
- define la variable objetivo a partir de la nota final `G3`
- permite mantener también la tarea original de regresión
- construye el atributo sensible (individual o interseccional)
- codifica las variables categóricas mediante one-hot y normaliza las numéricas
- devuelve X, y, sensitive y feature_names

Notas sobre el objetivo
-----------------------
El dataset original es de regresión. La variable objetivo original es:

- G3 : nota final del curso, entero en el rango [0, 20].

Para utilizarlo con LAMDA en tareas de clasificación, se discretiza por defecto
en tres clases ordinales mediante cuantiles:

- y = 0  -> low_grade
- y = 1  -> medium_grade
- y = 2  -> high_grade

Esta opción evita un fuerte desbalance de clases y permite aplicar las métricas
de fairness multiclase mediante un esquema one-vs-rest. También se ofrece una
opción binaria (aprobado/suspenso con corte habitual en 10) y la opción de
mantener la tarea de regresión original.

Notas sobre fuga de información
-------------------------------
Las notas parciales `G1` y `G2` presentan una correlación muy alta (> 0.9) con
`G3`. Para evitar que el problema se vuelva trivial, por defecto se excluyen de
las variables predictoras. El parámetro `include_partial_grades` permite
incluirlas explícitamente si se desea reproducir el escenario con notas previas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


SENSITIVE_MODE = Literal["sex", "address", "sex_address"]
TARGET_MODE = Literal["multiclass", "binary", "regression"]
MULTICLASS_STRATEGY = Literal["quantile", "fixed"]
BINARY_POSITIVE_CLASS = Literal["pass", "fail"]

BASE_DIR = Path(__file__).resolve().parents[3]


def _get_default_path() -> Path:
    """
    Devuelve la ruta por defecto al dataset Student Performance.

    Por defecto se utiliza el subconjunto de Portugués (student-por.csv) por
    disponer de más muestras (649 frente a 395 de Matemáticas), lo que ofrece
    más bloques para el esquema incremental.
    """
    return BASE_DIR / "data" / "raw" / "student" / "student-por.csv"


def _target_column() -> str:
    """
    Devuelve el nombre de la variable objetivo original.
    """
    return "G3"


def _partial_grade_columns() -> list[str]:
    """
    Devuelve las columnas de notas parciales con alta correlación con G3.

    Se excluyen por defecto de las variables predictoras para evitar fuga
    de información.
    """
    return ["G1", "G2"]


def _sensitive_source_columns() -> list[str]:
    """
    Devuelve las columnas originales usadas para construir el atributo sensible.

    Estas columnas se eliminan de las variables predictoras para que el atributo
    sensible no aparezca duplicado en X salvo decisión explícita del usuario.
    Nota: `sex` se mantiene en X por coherencia con el resto de loaders del
    proyecto (los atributos sensibles forman parte de X); aquí solo se listan
    a efectos de construcción del grupo sensible.
    """
    return ["sex", "address"]


def _build_sensitive(df: pd.DataFrame, mode: SENSITIVE_MODE) -> np.ndarray:
    """
    Construye el atributo sensible según el modo elegido.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    mode : {"sex", "address", "sex_address"}
        Modo de construcción del atributo sensible.
        - "sex"         : sexo del estudiante (F / M).
        - "address"     : tipo de zona de residencia (U=urbana, R=rural).
        - "sex_address" : combinación interseccional de ambos.

    Retorna
    -------
    np.ndarray
        Vector del atributo sensible.
    """
    sex_group = df["sex"].astype(str)
    address_group = df["address"].astype(str)

    if mode == "sex":
        sensitive = sex_group

    elif mode == "address":
        sensitive = address_group

    elif mode == "sex_address":
        sensitive = sex_group + "_" + address_group

    else:
        raise ValueError("Modo sensible no válido.")

    return sensitive.to_numpy()


def _student_feature_columns(
    df: pd.DataFrame,
    include_partial_grades: bool = False,
) -> list[str]:
    """
    Devuelve las columnas seleccionadas como features en Student Performance.

    Se elimina la variable objetivo (G3) y, por defecto, las notas parciales
    (G1, G2). Los atributos sensibles (sex, address) se mantienen en X por
    coherencia con el resto de loaders del proyecto.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    include_partial_grades : bool, default=False
        Si True, incluye G1 y G2 como variables predictoras.

    Retorna
    -------
    list[str]
        Lista de nombres de columnas usadas como features.
    """
    excluded_cols = {_target_column()}

    if not include_partial_grades:
        excluded_cols.update(_partial_grade_columns())

    return [col for col in df.columns if col not in excluded_cols]


def _build_multiclass_target(
    y_continuous: pd.Series,
    n_classes: int = 3,
    strategy: MULTICLASS_STRATEGY = "quantile",
    fixed_bins: list[float] | None = None,
) -> np.ndarray:
    """
    Construye una variable objetivo multiclase a partir de `G3`.

    Por defecto se usan cuantiles para crear clases ordinales balanceadas.

    Con n_classes = 3:
    - y = 0 -> low_grade
    - y = 1 -> medium_grade
    - y = 2 -> high_grade

    Parámetros
    ----------
    y_continuous : pd.Series
        Variable objetivo continua (nota G3).
    n_classes : int, default=3
        Número de clases.
    strategy : {"quantile", "fixed"}, default="quantile"
        Estrategia de discretización.
    fixed_bins : list[float] | None, default=None
        Cortes fijos si strategy="fixed".

    Retorna
    -------
    np.ndarray
        Vector objetivo discretizado.
    """
    if n_classes < 2:
        raise ValueError("n_classes debe ser mayor o igual que 2.")

    y_continuous = y_continuous.astype(float)

    if strategy == "quantile":
        labels = list(range(n_classes))

        y = pd.qcut(
            y_continuous,
            q=n_classes,
            labels=labels,
            duplicates="drop",
        )

        y_array = y.astype(int).to_numpy()

        # Con duplicates="drop", si la variable objetivo tiene valores muy
        # repetidos, pd.qcut puede generar menos clases de las solicitadas
        # sin lanzar error. Se avisa explícitamente para que el número real
        # de clases no pase inadvertido al configurar el experimento.
        n_obtained = len(np.unique(y_array))
        if n_obtained < n_classes:
            import warnings

            warnings.warn(
                f"La discretización por cuantiles produjo {n_obtained} clases "
                f"en lugar de las {n_classes} solicitadas, debido a valores "
                f"repetidos en la variable objetivo (duplicates='drop'). "
                f"Revise la distribución de G3 o considere strategy='fixed'.",
                stacklevel=2,
            )

        return y_array

    if strategy == "fixed":
        if fixed_bins is None:
            raise ValueError("fixed_bins debe informarse cuando strategy='fixed'.")

        if len(fixed_bins) != n_classes - 1:
            raise ValueError(
                "fixed_bins debe contener n_classes - 1 puntos de corte."
            )

        bins = [-np.inf, *fixed_bins, np.inf]
        labels = list(range(n_classes))

        y = pd.cut(
            y_continuous,
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        return y.astype(int).to_numpy()

    raise ValueError("Estrategia multiclase no válida.")


def _build_binary_target(
    y_continuous: pd.Series,
    threshold: float | None = None,
    positive_class: BINARY_POSITIVE_CLASS = "pass",
) -> np.ndarray:
    """
    Construye una variable objetivo binaria a partir de `G3`.

    El corte habitual de aprobado en la escala portuguesa (0-20) es 10.

    Parámetros
    ----------
    y_continuous : pd.Series
        Variable objetivo continua (nota G3).
    threshold : float | None, default=None
        Umbral de corte. Si es None, se usa 10 (aprobado).
    positive_class : {"pass", "fail"}, default="pass"
        Clase que se codifica como 1.

    Retorna
    -------
    np.ndarray
        Vector objetivo binario.
    """
    y_continuous = y_continuous.astype(float)

    if threshold is None:
        threshold = 10.0

    passed = y_continuous >= threshold

    if positive_class == "pass":
        return passed.astype(int).to_numpy()

    if positive_class == "fail":
        return (~passed).astype(int).to_numpy()

    raise ValueError("Clase positiva binaria no válida.")


def load_student_dataset(
    file_path: str | Path | None = None,
    sensitive_mode: SENSITIVE_MODE = "sex",
    target_mode: TARGET_MODE = "multiclass",
    n_classes: int = 3,
    multiclass_strategy: MULTICLASS_STRATEGY = "quantile",
    fixed_bins: list[float] | None = None,
    binary_threshold: float | None = None,
    binary_positive_class: BINARY_POSITIVE_CLASS = "pass",
    include_partial_grades: bool = False,
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carga y preprocesa el dataset Student Performance.

    Parámetros
    ----------
    file_path : str | Path | None, default=None
        Ruta al fichero CSV. Si es None, se usa la ruta por defecto
        (student-por.csv). El fichero usa separador ';'.
    sensitive_mode : {"sex", "address", "sex_address"}, default="sex"
        Modo de construcción del atributo sensible.
    target_mode : {"multiclass", "binary", "regression"}, default="multiclass"
        Modo de construcción de la variable objetivo.
    n_classes : int, default=3
        Número de clases si target_mode="multiclass".
    multiclass_strategy : {"quantile", "fixed"}, default="quantile"
        Estrategia para discretizar el objetivo multiclase.
    fixed_bins : list[float] | None, default=None
        Puntos de corte si multiclass_strategy="fixed".
    binary_threshold : float | None, default=None
        Umbral si target_mode="binary". Si es None, se usa 10 (aprobado).
    binary_positive_class : {"pass", "fail"}, default="pass"
        Clase positiva si target_mode="binary".
    include_partial_grades : bool, default=False
        Si True, incluye G1 y G2 como features. Por defecto se excluyen para
        evitar fuga de información (correlación > 0.9 con G3).
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
    (LamdaMinMaxNormalizer), ajustada únicamente sobre el conjunto de
    entrenamiento tras la partición train/test. Normalizar aquí, sobre el
    dataset completo antes de la partición, introduciría fuga de información
    del conjunto de test en el preprocesado.
    """
    if file_path is None:
        file_path = _get_default_path()
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {file_path}")

    # El dataset Student Performance utiliza ';' como separador.
    df = pd.read_csv(file_path, sep=";")

    if drop_na:
        df = df.dropna()

    if drop_duplicates:
        df = df.drop_duplicates()

    if df.empty:
        raise ValueError("Dataset vacío tras limpieza.")

    target_col = _target_column()
    y_continuous = pd.to_numeric(df[target_col], errors="coerce")

    # Se eliminan posibles filas con objetivo no numérico.
    valid_index = y_continuous.dropna().index
    df = df.loc[valid_index]
    y_continuous = y_continuous.loc[valid_index]

    if df.empty:
        raise ValueError("Dataset vacío tras limpiar el objetivo.")

    if target_mode == "regression":
        y = y_continuous.astype(float).to_numpy()

    elif target_mode == "binary":
        y = _build_binary_target(
            y_continuous=y_continuous,
            threshold=binary_threshold,
            positive_class=binary_positive_class,
        )

    elif target_mode == "multiclass":
        y = _build_multiclass_target(
            y_continuous=y_continuous,
            n_classes=n_classes,
            strategy=multiclass_strategy,
            fixed_bins=fixed_bins,
        )

    else:
        raise ValueError("Modo de objetivo no válido.")

    sensitive = _build_sensitive(df, sensitive_mode)

    feature_cols = _student_feature_columns(
        df,
        include_partial_grades=include_partial_grades,
    )
    X_df = df[feature_cols].copy()

    # Separación de variables categóricas y numéricas.
    categorical_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_df.select_dtypes(exclude=["object"]).columns.tolist()

    # Las variables numéricas se devuelven en su escala original, sin
    # normalizar. La normalización min-max de LAMDA se aplica en el
    # clasificador, ajustada solo sobre entrenamiento, para no introducir
    # fuga de información desde el test.
    X_num = X_df[numerical_cols].to_numpy(dtype=float)

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
    X, y, s, features = load_student_dataset()

    print("✔ Dataset Student Performance cargado correctamente")
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Shape sensitive:", s.shape)
    print("Nº features:", len(features))
    print("Clases en y:", np.unique(y, return_counts=True))
    print("Grupos sensibles:", np.unique(s, return_counts=True))
