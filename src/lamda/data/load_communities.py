"""
Carga y preprocesado del dataset Communities and Crime para experimentos de fairness con LAMDA.

Este módulo:
- carga el dataset Communities and Crime
- limpia valores faltantes
- define la variable objetivo multiclase a partir de `ViolentCrimesPerPop`
- permite mantener también la tarea original de regresión
- construye el atributo sensible/interseccional
- normaliza las variables numéricas
- devuelve X, y, sensitive y feature_names

Notas sobre el objetivo
-----------------------
El dataset original es de regresión. La variable objetivo original es:

- ViolentCrimesPerPop

Para utilizarlo con LAMDA en tareas de clasificación, se discretiza por defecto
en tres clases ordinales mediante cuantiles:

- y = 0  -> low_crime
- y = 1  -> medium_crime
- y = 2  -> high_crime

Esta opción evita un fuerte desbalance de clases y permite aplicar las métricas
de fairness multiclase mediante un esquema one-vs-rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


SENSITIVE_MODE = Literal["black", "race", "income", "race_income"]
TARGET_MODE = Literal["multiclass", "binary", "regression"]
MULTICLASS_STRATEGY = Literal["quantile", "fixed"]
BINARY_POSITIVE_CLASS = Literal["low_crime", "high_crime"]

BASE_DIR = Path(__file__).resolve().parents[3]


def _get_default_path() -> Path:
    """
    Devuelve la ruta por defecto al dataset Communities and Crime.
    """
    return BASE_DIR / "data" / "raw" / "communities" / "communities.data"


def _communities_column_names() -> list[str]:
    """
    Devuelve los nombres de columnas del dataset Communities and Crime.

    Los nombres corresponden al fichero communities.names de UCI.
    """
    return [
        "state",
        "county",
        "community",
        "communityname",
        "fold",
        "population",
        "householdsize",
        "racepctblack",
        "racePctWhite",
        "racePctAsian",
        "racePctHisp",
        "agePct12t21",
        "agePct12t29",
        "agePct16t24",
        "agePct65up",
        "numbUrban",
        "pctUrban",
        "medIncome",
        "pctWWage",
        "pctWFarmSelf",
        "pctWInvInc",
        "pctWSocSec",
        "pctWPubAsst",
        "pctWRetire",
        "medFamInc",
        "perCapInc",
        "whitePerCap",
        "blackPerCap",
        "indianPerCap",
        "AsianPerCap",
        "OtherPerCap",
        "HispPerCap",
        "NumUnderPov",
        "PctPopUnderPov",
        "PctLess9thGrade",
        "PctNotHSGrad",
        "PctBSorMore",
        "PctUnemployed",
        "PctEmploy",
        "PctEmplManu",
        "PctEmplProfServ",
        "PctOccupManu",
        "PctOccupMgmtProf",
        "MalePctDivorce",
        "MalePctNevMarr",
        "FemalePctDiv",
        "TotalPctDiv",
        "PersPerFam",
        "PctFam2Par",
        "PctKids2Par",
        "PctYoungKids2Par",
        "PctTeen2Par",
        "PctWorkMomYoungKids",
        "PctWorkMom",
        "NumIlleg",
        "PctIlleg",
        "NumImmig",
        "PctImmigRecent",
        "PctImmigRec5",
        "PctImmigRec8",
        "PctImmigRec10",
        "PctRecentImmig",
        "PctRecImmig5",
        "PctRecImmig8",
        "PctRecImmig10",
        "PctSpeakEnglOnly",
        "PctNotSpeakEnglWell",
        "PctLargHouseFam",
        "PctLargHouseOccup",
        "PersPerOccupHous",
        "PersPerOwnOccHous",
        "PersPerRentOccHous",
        "PctPersOwnOccup",
        "PctPersDenseHous",
        "PctHousLess3BR",
        "MedNumBR",
        "HousVacant",
        "PctHousOccup",
        "PctHousOwnOcc",
        "PctVacantBoarded",
        "PctVacMore6Mos",
        "MedYrHousBuilt",
        "PctHousNoPhone",
        "PctWOFullPlumb",
        "OwnOccLowQuart",
        "OwnOccMedVal",
        "OwnOccHiQuart",
        "RentLowQ",
        "RentMedian",
        "RentHighQ",
        "MedRent",
        "MedRentPctHousInc",
        "MedOwnCostPctInc",
        "MedOwnCostPctIncNoMtg",
        "NumInShelters",
        "NumStreet",
        "PctForeignBorn",
        "PctBornSameState",
        "PctSameHouse85",
        "PctSameCity85",
        "PctSameState85",
        "LemasSwornFT",
        "LemasSwFTPerPop",
        "LemasSwFTFieldOps",
        "LemasSwFTFieldPerPop",
        "LemasTotalReq",
        "LemasTotReqPerPop",
        "PolicReqPerOffic",
        "PolicPerPop",
        "RacialMatchCommPol",
        "PctPolicWhite",
        "PctPolicBlack",
        "PctPolicHisp",
        "PctPolicAsian",
        "PctPolicMinor",
        "OfficAssgnDrugUnits",
        "NumKindsDrugsSeiz",
        "PolicAveOTWorked",
        "LandArea",
        "PopDens",
        "PctUsePubTrans",
        "PolicCars",
        "PolicOperBudg",
        "LemasPctPolicOnPatr",
        "LemasGangUnitDeploy",
        "LemasPctOfficDrugUn",
        "PolicBudgPerPop",
        "ViolentCrimesPerPop",
    ]


def _id_columns() -> list[str]:
    """
    Devuelve las columnas identificativas que no se usan como variables predictoras.

    Estas columnas identifican el estado, condado, comunidad y fold, pero no deben
    utilizarse como descriptores del modelo.
    """
    return [
        "state",
        "county",
        "community",
        "communityname",
        "fold",
    ]


def _target_column() -> str:
    """
    Devuelve el nombre de la variable objetivo original.
    """
    return "ViolentCrimesPerPop"


def _build_race_group(df: pd.DataFrame) -> pd.Series:
    """
    Construye el grupo racial predominante de cada comunidad.

    Se toma como grupo racial aquel con mayor porcentaje entre:
    - Black
    - White
    - Asian
    - Hispanic

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Retorna
    -------
    pd.Series
        Serie con el grupo racial predominante.
    """
    race_cols = {
        "Black": "racepctblack",
        "White": "racePctWhite",
        "Asian": "racePctAsian",
        "Hispanic": "racePctHisp",
    }

    race_df = df[list(race_cols.values())].copy()
    race_df = race_df.apply(pd.to_numeric, errors="coerce")

    inverse_map = {v: k for k, v in race_cols.items()}

    predominant_race = race_df.idxmax(axis=1).map(inverse_map)

    return predominant_race.astype(str)


def _build_black_group(df: pd.DataFrame, threshold: float = 0.06) -> pd.Series:
    """
    Construye el grupo sensible binario "Black" / "NonBlack" siguiendo la
    convención de la literatura de fairness sobre este dataset.

    Le Quy et al. (2022), a partir de Kamishima et al. (2012) y Kamiran et al.
    (2013), derivan un atributo binario umbralizando `racepctblack` (porcentaje
    de población afroamericana de la comunidad) en 0.06: las comunidades con un
    porcentaje superior al umbral se etiquetan como "Black" y el resto como
    "NonBlack". Se adopta esta binarización exacta para que la disparidad de
    partida sea comparable, en orden de magnitud, con la reportada en dicha
    literatura.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    threshold : float, default=0.06
        Umbral sobre `racepctblack`. El valor 0.06 reproduce la convención
        de la literatura de referencia.

    Retorna
    -------
    pd.Series
        Serie con el grupo binario {"Black", "NonBlack"}.
    """
    racepctblack = pd.to_numeric(df["racepctblack"], errors="coerce")

    black_group = np.where(racepctblack > threshold, "Black", "NonBlack")

    return pd.Series(black_group, index=df.index).astype(str)


def _build_income_group(df: pd.DataFrame) -> pd.Series:
    """
    Construye el grupo de nivel de ingresos a partir de `medIncome`.

    Se discretiza `medIncome` en tres grupos mediante terciles:
    - LowIncome
    - MediumIncome
    - HighIncome

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.

    Retorna
    -------
    pd.Series
        Serie con el grupo de ingresos.
    """
    income = pd.to_numeric(df["medIncome"], errors="coerce")

    income_group = pd.qcut(
        income,
        q=3,
        labels=["LowIncome", "MediumIncome", "HighIncome"],
        duplicates="drop",
    )

    return income_group.astype(str)


def _build_sensitive(df: pd.DataFrame, mode: SENSITIVE_MODE) -> np.ndarray:
    """
    Construye el atributo sensible según el modo elegido.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del dataset.
    mode : {"race", "income", "race_income"}
        Modo de construcción del atributo sensible.

    Retorna
    -------
    np.ndarray
        Vector del atributo sensible.
    """
    if mode == "black":
        # El modo "black" no requiere calcular los grupos de raza predominante
        # ni de ingresos; se construye directamente por umbral sobre
        # racepctblack, según la convención de Le Quy et al. (2022).
        return _build_black_group(df).to_numpy()

    race_group = _build_race_group(df)
    income_group = _build_income_group(df)

    if mode == "race":
        sensitive = race_group

    elif mode == "income":
        sensitive = income_group

    elif mode == "race_income":
        sensitive = race_group.astype(str) + "_" + income_group.astype(str)

    else:
        raise ValueError("Modo sensible no válido.")

    return sensitive.to_numpy()


def _communities_feature_columns() -> list[str]:
    """
    Devuelve las columnas seleccionadas como features en Communities and Crime.

    Se eliminan las columnas identificativas y la variable objetivo.
    """
    excluded_cols = set(_id_columns() + [_target_column()])

    return [
        col
        for col in _communities_column_names()
        if col not in excluded_cols
    ]


def _build_multiclass_target(
    y_continuous: pd.Series,
    n_classes: int = 3,
    strategy: MULTICLASS_STRATEGY = "quantile",
    fixed_bins: list[float] | None = None,
) -> np.ndarray:
    """
    Construye una variable objetivo multiclase a partir de `ViolentCrimesPerPop`.

    Por defecto se usan cuantiles para crear clases ordinales balanceadas.

    Con n_classes = 3:
    - y = 0 -> low_crime
    - y = 1 -> medium_crime
    - y = 2 -> high_crime

    Parámetros
    ----------
    y_continuous : pd.Series
        Variable objetivo continua.
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

        return y.astype(int).to_numpy()

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
    positive_class: BINARY_POSITIVE_CLASS = "low_crime",
) -> np.ndarray:
    """
    Construye una variable objetivo binaria a partir de `ViolentCrimesPerPop`.

    Parámetros
    ----------
    y_continuous : pd.Series
        Variable objetivo continua.
    threshold : float | None, default=None
        Umbral de corte. Si es None, se usa la mediana.
    positive_class : {"low_crime", "high_crime"}, default="low_crime"
        Clase que se codifica como 1.

    Retorna
    -------
    np.ndarray
        Vector objetivo binario.
    """
    y_continuous = y_continuous.astype(float)

    if threshold is None:
        threshold = y_continuous.median()

    high_crime = y_continuous > threshold

    if positive_class == "high_crime":
        return high_crime.astype(int).to_numpy()

    if positive_class == "low_crime":
        return (~high_crime).astype(int).to_numpy()

    raise ValueError("Clase positiva binaria no válida.")


def load_communities_dataset(
    file_path: str | Path | None = None,
    sensitive_mode: SENSITIVE_MODE = "black",
    target_mode: TARGET_MODE = "multiclass",
    n_classes: int = 3,
    multiclass_strategy: MULTICLASS_STRATEGY = "quantile",
    fixed_bins: list[float] | None = None,
    binary_threshold: float | None = None,
    binary_positive_class: BINARY_POSITIVE_CLASS = "low_crime",
    fill_na: bool = True,
    drop_na: bool = False,
    drop_duplicates: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carga y preprocesa el dataset Communities and Crime.

    Parámetros
    ----------
    file_path : str | Path | None, default=None
        Ruta al fichero. Si es None, se usa la ruta por defecto.
    sensitive_mode : {"black", "race", "income", "race_income"}, default="black"
        Modo de construcción del atributo sensible. Por defecto "black"
        (binario Black/NonBlack por umbral 0.06 sobre racepctblack), que
        reproduce la convención de Le Quy et al. (2022) y permite comparar
        la disparidad de partida con la literatura. Los modos "race" (raza
        predominante entre cuatro grupos), "income" (terciles de renta) y
        "race_income" (interseccional) quedan disponibles para análisis
        adicionales, pero no son comparables con la literatura de referencia.
    target_mode : {"multiclass", "binary", "regression"}, default="multiclass"
        Modo de construcción de la variable objetivo.
    n_classes : int, default=3
        Número de clases si target_mode="multiclass".
    multiclass_strategy : {"quantile", "fixed"}, default="quantile"
        Estrategia para discretizar el objetivo multiclase.
    fixed_bins : list[float] | None, default=None
        Puntos de corte si multiclass_strategy="fixed".
    binary_threshold : float | None, default=None
        Umbral si target_mode="binary". Si es None, se usa la mediana.
    binary_positive_class : {"low_crime", "high_crime"}, default="low_crime"
        Clase positiva si target_mode="binary".
    fill_na : bool, default=True
        Si True, imputa valores faltantes con la mediana.
    drop_na : bool, default=False
        Si True, elimina filas con valores faltantes.
        No se recomienda como opción por defecto porque este dataset tiene muchos nulos.
    drop_duplicates : bool, default=True
        Si True, elimina filas duplicadas.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
        X, y, sensitive y feature_names.

    Notas sobre normalización
    -------------------------
    Este loader NO normaliza las variables. La normalización min-max propia
    de LAMDA se realiza dentro del clasificador (LamdaMinMaxNormalizer),
    ajustada únicamente sobre el conjunto de entrenamiento tras la partición
    train/test. Normalizar aquí, sobre el dataset completo antes de la
    partición, introduciría fuga de información del conjunto de test. Nótese
    que la mayoría de atributos de este dataset ya vienen normalizados en
    [0, 1] en la fuente original; aun así, la normalización se delega en el
    clasificador para mantener un tratamiento homogéneo con el resto de
    loaders y evitar cualquier ajuste de escala previo a la partición.
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
        names=_communities_column_names(),
        na_values="?",
    )

    if drop_duplicates:
        df = df.drop_duplicates()

    feature_cols = _communities_feature_columns()
    target_col = _target_column()

    X_df = df[feature_cols].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    y_continuous = pd.to_numeric(df[target_col], errors="coerce")

    if drop_na:
        valid_index = X_df.dropna().index
        X_df = X_df.loc[valid_index]
        y_continuous = y_continuous.loc[valid_index]
        df = df.loc[valid_index]

    if fill_na:
        X_df = X_df.fillna(X_df.median(numeric_only=True))

    X_df = X_df.fillna(0.0)

    if df.empty or X_df.empty:
        raise ValueError("Dataset vacío tras limpieza.")

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

    # Las variables se devuelven sin normalizar; la normalización min-max de
    # LAMDA se aplica en el clasificador, ajustada solo sobre entrenamiento,
    # para no introducir fuga de información desde el test.
    X = X_df.to_numpy(dtype=float)

    feature_names = X_df.columns.tolist()

    return X, y, sensitive, feature_names


def describe_communities_target(
    file_path: str | Path | None = None,
    n_classes: int = 3,
) -> pd.DataFrame:
    """
    Resume distintas formas de discretizar la variable objetivo.
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
        names=_communities_column_names(),
        na_values="?",
    )

    y_continuous = pd.to_numeric(df[_target_column()], errors="coerce")

    rows = []

    rows.append(
        {
            "mode": "regression",
            "class": "continuous",
            "count": int(y_continuous.notna().sum()),
            "percentage": 100.0,
            "info": (
                f"min={y_continuous.min():.4f}, "
                f"median={y_continuous.median():.4f}, "
                f"mean={y_continuous.mean():.4f}, "
                f"max={y_continuous.max():.4f}"
            ),
        }
    )

    y_binary_median = _build_binary_target(
        y_continuous=y_continuous,
        threshold=None,
        positive_class="low_crime",
    )

    for class_value, class_name in [(0, "high_crime"), (1, "low_crime")]:
        count = int((y_binary_median == class_value).sum())
        rows.append(
            {
                "mode": "binary_median",
                "class": class_name,
                "count": count,
                "percentage": round(100 * count / len(y_binary_median), 2),
                "info": f"threshold=median={y_continuous.median():.4f}",
            }
        )

    y_binary_07 = _build_binary_target(
        y_continuous=y_continuous,
        threshold=0.7,
        positive_class="high_crime",
    )

    for class_value, class_name in [(0, "low_crime"), (1, "high_crime")]:
        count = int((y_binary_07 == class_value).sum())
        rows.append(
            {
                "mode": "binary_threshold_0.7",
                "class": class_name,
                "count": count,
                "percentage": round(100 * count / len(y_binary_07), 2),
                "info": "threshold=0.7",
            }
        )

    y_multiclass = _build_multiclass_target(
        y_continuous=y_continuous,
        n_classes=n_classes,
        strategy="quantile",
    )

    class_names = {
        0: "low_crime",
        1: "medium_crime",
        2: "high_crime",
    }

    quantiles = y_continuous.quantile(
        [i / n_classes for i in range(1, n_classes)]
    )

    info = ", ".join(
        f"q{int(q * 100)}={value:.4f}"
        for q, value in quantiles.items()
    )

    for class_value in sorted(np.unique(y_multiclass)):
        count = int((y_multiclass == class_value).sum())
        rows.append(
            {
                "mode": f"multiclass_quantile_{n_classes}",
                "class": class_names.get(int(class_value), f"class_{class_value}"),
                "count": count,
                "percentage": round(100 * count / len(y_multiclass), 2),
                "info": info,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    X, y, s, features = load_communities_dataset()

    print("✔ Dataset Communities and Crime cargado correctamente")
    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Shape sensitive:", s.shape)
    print("Nº features:", len(features))
    print("Clases en y:", np.unique(y))
    print("Distribución de y:")
    print(pd.Series(y).value_counts().sort_index())
    print("Grupos sensibles:", np.unique(s))
    print()
    print("Resumen de discretización del target:")
    print(describe_communities_target())