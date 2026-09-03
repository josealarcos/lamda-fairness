"""
Configuración centralizada de los experimentos de detección y mitigación.

Este módulo reúne en un único lugar la configuración de los seis conjuntos de
datos empleados en la fase experimental del trabajo, de modo que los notebooks
de experimentos (detección, mitigación y agregación) no dispersen parámetros ni
dupliquen decisiones. Cada dataset se describe mediante:

- el loader que lo carga,
- el atributo sensible por defecto (alineado con la literatura de referencia
  para maximizar la comparabilidad de la fase de detección),
- el tipo de tarea (binaria o multiclase),
- los argumentos específicos de carga,
- y los parámetros del esquema incremental por bloques.

Composición de datasets
-----------------------
Binarios (comparables con Le Quy et al., 2022, por magnitud de disparidad):
- adult   : ingresos, atributo sensible sex.
- dutch   : ocupación (alto/bajo nivel), atributo sensible sex.
- compas  : reincidencia, atributo sensible race (favorable = no reincidir).

Multiclase:
- oulad       : resultado académico (4 clases), atributo sensible gender.
- communities : criminalidad discretizada por cuantiles, atributo sensible black.
- student     : rendimiento discretizado a multiclase ordinal, atributo sensible sex.

Nota sobre la composición (actualización 2026-08-09)
----------------------------------------------------
Bank Marketing se retira del panel activo y se sustituye por Dutch Census: Bank
tiene la clase objetivo muy desbalanceada (~88/12) y LAMDA clásico no la supera
por encima del baseline trivial, por lo que su disparidad no es interpretable.
Dutch está equilibrado (~48/52), es comparable en Le Quy et al. (2022) con
SPD 0.3568 y presenta disparidad marcada, adecuada para evaluar la mitigación.
El loader load_bank sigue disponible para el análisis de la limitación de LAMDA
bajo desbalance. OULAD queda pendiente de revisión (rendimiento por debajo del
baseline trivial por pobreza de features).

Nota sobre el perfil de métricas
--------------------------------
El perfil de métricas (qué métricas intervienen en D_g y en el reporte) se
deriva del número de clases mediante el módulo metric_profiles: perfil binario
(SPD, EOD, DI) para los datasets binarios y perfil multiclase (SPD, EOD sin DI,
con reporte adicional de SPD multiclase y MEO) para los multiclase. La
configuración de este módulo indica el tipo de tarea, y el runner construye el
perfil correspondiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from lamda.data.load_adult import load_adult_dataset
from lamda.data.load_dutch import load_dutch_dataset
from lamda.data.load_compas import load_compas_dataset
from lamda.data.load_communities import load_communities_dataset
from lamda.data.load_oulad import load_oulad_dataset
from lamda.data.load_student import load_student_dataset

from lamda.fairness.disparity import FairnessThresholds, FairnessWeights


TaskKind = Literal["binary", "multiclass"]


# =========================================================
# DESCRIPCIÓN DE UN DATASET
# =========================================================

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuración de un conjunto de datos para los experimentos.

    Atributos
    ---------
    name : str
        Identificador del dataset (usado en nombres de ficheros de resultados).
    loader : Callable
        Función loader que devuelve (X, y, sensitive, feature_names).
    sensitive_mode : str
        Modo de atributo sensible por defecto, alineado con la literatura de
        referencia para la comparación de detección.
    task : {"binary", "multiclass"}
        Tipo de tarea. Determina el perfil de métricas aplicado.
    loader_kwargs : dict
        Argumentos específicos que se pasan al loader (limpieza, construcción
        del objetivo, etc.).
    display_name : str
        Nombre legible para títulos de figuras y tablas.
    sensitive_feature_patterns : dict[str, tuple[str, ...]]
        Para cada modo de atributo sensible, los nombres (o prefijos de nombre)
        de las columnas de la matriz de descriptores que codifican ese atributo.
        Permite ejecutar el experimento excluyendo el atributo sensible de X,
        como análisis de ablación. Los patrones son explícitos por conjunto
        porque la codificación difiere entre loaders: los atributos categóricos
        se expanden mediante one-hot con el prefijo del nombre original, mientras
        que en Communities and Crime el grupo sensible se deriva por umbral de
        una variable continua, de modo que la columna a excluir no coincide con
        el nombre del modo.
    """

    name: str
    loader: Callable[..., tuple]
    sensitive_mode: str
    task: TaskKind
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    sensitive_feature_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)


# =========================================================
# CONFIGURACIÓN DE LOS SEIS DATASETS
# =========================================================

DATASETS: dict[str, DatasetConfig] = {
    # -------------------- BINARIOS --------------------
    "adult": DatasetConfig(
        name="adult",
        loader=load_adult_dataset,
        sensitive_mode="sex",
        task="binary",
        loader_kwargs={
            "drop_na": True,
            "drop_duplicates": True,
        },
        display_name="Adult",
        sensitive_feature_patterns={"sex": ("sex_",), "race": ("race_",)},
    ),
    "dutch": DatasetConfig(
        name="dutch",
        loader=load_dutch_dataset,
        sensitive_mode="sex",
        task="binary",
        loader_kwargs={
            "drop_na": True,
            # No se eliminan duplicados: al tratarse de registros censales
            # descritos exclusivamente mediante códigos categóricos, dos filas
            # idénticas corresponden a individuos distintos con el mismo perfil
            # y no a una duplicación de la medida. Conservarlas preserva además
            # la muestra de 60.420 registros sobre la que Le Quy et al. (2022)
            # calculan las cifras de disparidad usadas como referencia externa,
            # lo que es condición para la comparabilidad de la detección.
            "drop_duplicates": False,
        },
        display_name="Dutch Census",
        sensitive_feature_patterns={"sex": ("sex_",)},
    ),
    "compas": DatasetConfig(
        name="compas",
        loader=load_compas_dataset,
        sensitive_mode="race",
        task="binary",
        loader_kwargs={
            "drop_na": True,
            "drop_duplicates": True,
            "apply_filters": True,
            "binarize_race": True,
            "favorable_no_recid": True,
        },
        display_name="COMPAS",
        sensitive_feature_patterns={"race": ("race_",)},
    ),
    # -------------------- MULTICLASE --------------------
    "oulad": DatasetConfig(
        name="oulad",
        loader=load_oulad_dataset,
        sensitive_mode="gender",
        task="multiclass",
        loader_kwargs={
            "target_mode": "multiclass",
            "drop_na": True,
            "drop_duplicates": True,
        },
        display_name="OULAD",
        sensitive_feature_patterns={
            "gender": ("gender_",),
            "disability": ("disability_",),
            "age_band": ("age_band_",),
        },
    ),
    "communities": DatasetConfig(
        name="communities",
        loader=load_communities_dataset,
        sensitive_mode="black",
        task="multiclass",
        loader_kwargs={
            "target_mode": "multiclass",
            "n_classes": 3,
            "multiclass_strategy": "quantile",
            "fill_na": True,
            "drop_na": False,
            "drop_duplicates": True,
        },
        display_name="Communities and Crime",
        # El grupo sensible se deriva umbralizando racepctblack; blackPerCap es
        # una variable de renta y no debe excluirse al retirar el atributo.
        sensitive_feature_patterns={"black": ("racepctblack",)},
    ),
    "student": DatasetConfig(
        name="student",
        loader=load_student_dataset,
        sensitive_mode="sex",
        task="multiclass",
        loader_kwargs={
            "target_mode": "multiclass",
            "n_classes": 3,
            "multiclass_strategy": "quantile",
            "include_partial_grades": False,
            "drop_na": True,
            "drop_duplicates": True,
        },
        display_name="Student Performance",
        sensitive_feature_patterns={"sex": ("sex_",)},
    ),
}


# =========================================================
# PARÁMETROS COMUNES DEL EXPERIMENTO
# =========================================================

# Partición train/test.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Parámetros del modelo LAMDA.
ALPHA = 0.65
AGGREGATION_OPERATOR = "minmax"  # operador de referencia estable

# Umbrales de equidad (convenciones de práctica).
FAIRNESS_THRESHOLDS = FairnessThresholds(spd=0.1, eod=0.1, di=0.8)

# Parámetros de Fair-GAD.
LAMBDA_FAIR = 0.1

# Parámetros de Fair-MAD.
ETA_FAIR = 10
TAU_J = 0.0
UPDATE_EVERY_N_BLOCKS = 1

# =========================================================
# ESQUEMA INCREMENTAL POR BLOQUES
# =========================================================
#
# El tamaño de bloque se deriva del tamaño del conjunto de entrenamiento en
# lugar de fijarse en un valor constante. Con un tamaño de bloque fijo, el
# número de actualizaciones del estado de equidad depende del tamaño del
# conjunto, de modo que el mecanismo se adapta muchas más veces en los
# conjuntos grandes que en los pequeños y las comparaciones entre conjuntos
# quedan condicionadas por esa diferencia. Fijar en su lugar el número de
# bloques homogeneiza el número de adaptaciones.
#
# Tiene además una consecuencia computacional relevante. El coste de
# reconstruir la regularización sobre el histórico acumulado crece con el
# producto del número de individuos por el número de bloques; al mantener
# constante el segundo, el coste pasa a ser aproximadamente lineal en el
# número de individuos en lugar de cuadrático.
#
# El tamaño mínimo evita bloques demasiado pequeños en los conjuntos de menor
# tamaño, en los que un bloque reducido produciría estimaciones de disparidad
# inestables en las primeras iteraciones.
BLOQUES_OBJETIVO = 20
TAMANO_BLOQUE_MINIMO = 64

# Tamaño de bloque de referencia, conservado para reproducir configuraciones
# anteriores y para su uso directo cuando no se desee derivarlo del tamaño.
BLOCK_SIZE = 128
# Soportes mínimos de referencia. Su finalidad es descartar estimaciones de
# disparidad basadas en un número de observaciones demasiado pequeño para ser
# fiable. No proceden de un valor convencional de la literatura, que no lo fija:
# son criterio propio, coherente con la advertencia de las Uniform Guidelines on
# Employee Selection Procedures (29 CFR 1607.4 D) sobre diferencias basadas en
# números pequeños.
MIN_GROUP_SIZE = 50
MIN_POSITIVE_SIZE = 25

# Parámetros del criterio automático de soportes.
PISO_SOPORTE_GRUPO = 20    # cota inferior del soporte por grupo
PISO_SOPORTE_CLASE = 8     # cota inferior del soporte por par grupo-clase
FRACCION_BLOQUE = 0.5      # fracción del bloque que limita el soporte por grupo
DEFAULT_GROUP_DISPARITY = 0.0
USE_CUMULATIVE_HISTORY = True

# Directorio donde se persisten los resultados por dataset, para que el
# notebook de agregación pueda leerlos.
RESULTS_DIR = "resultados"


# =========================================================
# UTILIDADES
# =========================================================

def tamano_bloque(
    n_train: int,
    bloques_objetivo: int = BLOQUES_OBJETIVO,
    minimo: int = TAMANO_BLOQUE_MINIMO,
) -> int:
    """
    Deriva el tamaño de bloque a partir del tamaño del conjunto de
    entrenamiento, de modo que el número de bloques sea aproximadamente el
    mismo en todos los conjuntos.

    Parámetros
    ----------
    n_train : int
        Número de individuos del conjunto de entrenamiento.
    bloques_objetivo : int
        Número de bloques deseado.
    minimo : int
        Tamaño mínimo de bloque. Prevalece sobre el número objetivo en los
        conjuntos de menor tamaño, en los que respetar el objetivo produciría
        bloques demasiado pequeños para estimar la disparidad de forma estable.

    Retorna
    -------
    int
        Tamaño de bloque aplicable.
    """
    if n_train <= 0:
        raise ValueError("n_train debe ser mayor que 0.")
    if bloques_objetivo <= 0:
        raise ValueError("bloques_objetivo debe ser mayor que 0.")

    derivado = (n_train + bloques_objetivo - 1) // bloques_objetivo
    return max(minimo, derivado)


def soportes_minimos(
    n_train: int,
    n_classes: int = 2,
    block_size: int | None = None,
    modo: str = "auto",
    proporcion_grupo: float = 0.02,
    proporcion_clase: float = 0.01,
) -> tuple[int, int]:
    """
    Devuelve los soportes mínimos por grupo y por par grupo-clase.

    Modos disponibles
    -----------------
    "auto" (recomendado)
        El soporte por grupo parte del valor de referencia y se limita a una
        fracción del tamaño de bloque, de modo que en los conjuntos pequeños,
        cuyos bloques son necesariamente reducidos, el filtro no inhabilite las
        primeras iteraciones del esquema incremental. Sin esta limitación, un
        umbral constante deja fuera una proporción mucho mayor del calendario en
        los conjuntos pequeños que en los grandes, lo que introduce una
        asimetría ajena al fenómeno estudiado.

        El soporte por par grupo-clase se obtiene dividiendo el anterior entre
        el número de clases. Bajo la descomposición uno contra el resto, la
        cantidad esperada de observaciones de una clase dentro de un grupo es
        aproximadamente el tamaño del grupo dividido entre el número de clases,
        de modo que un valor constante resultaría tanto más exigente cuanto
        mayor fuese el número de clases. Con dos clases se recupera el valor de
        referencia de 25.

        Ambos quedan acotados inferiormente para preservar la fiabilidad de la
        estimación.

    "absoluto"
        Valores de referencia constantes, con independencia del conjunto.

    "proporcional"
        Proporciones del tamaño de entrenamiento, con los valores de referencia
        como cota inferior. Se conserva para poder contrastar el criterio.

    Parámetros
    ----------
    n_train : int
        Número de individuos del conjunto de entrenamiento.
    n_classes : int, default=2
        Número de clases del problema.
    block_size : int | None, default=None
        Tamaño de bloque aplicable. Necesario en el modo automático.
    modo : {"auto", "absoluto", "proporcional"}
        Criterio de fijación.
    proporcion_grupo, proporcion_clase : float
        Proporciones empleadas en el modo proporcional.

    Retorna
    -------
    tuple[int, int]
        Soporte mínimo por grupo y soporte mínimo por par grupo-clase.
    """
    if n_classes < 2:
        raise ValueError("n_classes debe ser al menos 2.")

    if modo == "absoluto":
        return MIN_GROUP_SIZE, MIN_POSITIVE_SIZE

    if modo == "proporcional":
        return (
            max(MIN_GROUP_SIZE, int(round(proporcion_grupo * n_train))),
            max(MIN_POSITIVE_SIZE, int(round(proporcion_clase * n_train))),
        )

    if modo == "auto":
        if block_size is None:
            raise ValueError("El modo automático requiere block_size.")
        grupo = min(
            MIN_GROUP_SIZE,
            max(PISO_SOPORTE_GRUPO, int(round(FRACCION_BLOQUE * block_size))),
        )
        clase = max(PISO_SOPORTE_CLASE, int(round(grupo / n_classes)))
        return grupo, clase

    raise ValueError("modo debe ser 'auto', 'absoluto' o 'proporcional'.")


def get_dataset_config(name: str) -> DatasetConfig:
    """
    Recupera la configuración de un dataset por su identificador.

    Parámetros
    ----------
    name : str
        Identificador del dataset.

    Retorna
    -------
    DatasetConfig
        Configuración del dataset.

    Lanza
    ------
    ValueError
        Si el dataset no está registrado.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Dataset no soportado: {name}. "
            f"Opciones: {sorted(DATASETS.keys())}"
        )
    return DATASETS[name]


def binary_datasets() -> list[str]:
    """Devuelve los identificadores de los datasets binarios."""
    return [k for k, v in DATASETS.items() if v.task == "binary"]


def multiclass_datasets() -> list[str]:
    """Devuelve los identificadores de los datasets multiclase."""
    return [k for k, v in DATASETS.items() if v.task == "multiclass"]


def all_datasets() -> list[str]:
    """Devuelve los identificadores de los seis datasets, binarios primero."""
    return binary_datasets() + multiclass_datasets()
