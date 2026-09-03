"""
Motor de ejecución de los experimentos de detección y mitigación.

Este módulo encapsula la lógica de un experimento completo para un conjunto de
datos: carga, partición, entrenamiento de los tres modelos (LAMDA clásico,
Fair-GAD, Fair-MAD), evaluación de detección y de mitigación, evolución por
bloques y persistencia de resultados. La lógica se ha extraído del flujo de
trabajo validado en notebooks previos.

Principio de diseño
-------------------
Ninguna función fija valores de parámetros del experimento. Todos (alpha,
operador, lambda, eta, umbrales, tamaño de bloque, etc.) se reciben como
argumentos, de modo que el notebook mantiene el control total de la
configuración. El módulo config proporciona valores por defecto que el notebook
puede sobrescribir.

Separación detección / mitigación
----------------------------------
- La DETECCIÓN caracteriza la disparidad de partida del LAMDA clásico,
  incluyendo la evolución por bloques.
- La MITIGACIÓN compara los tres modelos sobre el conjunto de test con los
  mecanismos de equidad congelados.

Ambas fases comparten el entrenamiento base, por lo que run_experiment ejecuta
todo y devuelve un objeto con los resultados de las dos fases; los notebooks de
detección y de mitigación consumen las partes que les corresponden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lamda.models.lamda_classifier import LamdaClassifier
from lamda.models.lamda_fair_gad_classifier import LamdaFairGADClassifier
from lamda.models.lamda_fair_mad_classifier import LamdaFairMADClassifier

from lamda.fairness.disparity import FairnessThresholds, FairnessWeights
from lamda.fairness.metric_profiles import build_metric_profile

from lamda.experiments import reporting
from lamda.experiments import config as cfg


# =========================================================
# RESULTADO DE UN EXPERIMENTO
# =========================================================

@dataclass
class ExperimentResult:
    """
    Contenedor de los resultados de un experimento completo para un dataset.

    Atributos
    ---------
    dataset_name : str
        Identificador del dataset.
    task : str
        Tipo de tarea ("binary" o "multiclass").
    predictions : dict
        Predicciones sobre test de cada modelo.
    detection_table : pd.DataFrame
        Tabla de detección (métricas del LAMDA clásico sobre test).
    mitigation_table : pd.DataFrame
        Tabla de mitigación (tres modelos comparados sobre test).
    evolution_gad : pd.DataFrame
        Evolución por bloques de Fair-GAD durante el entrenamiento.
    evolution_mad : pd.DataFrame
        Evolución por bloques de Fair-MAD durante el entrenamiento.
    descriptor_weights : pd.DataFrame
        Pesos w_j y disparidades D_j por descriptor (Fair-MAD).
    metadata : dict
        Configuración usada (para trazabilidad).
    """

    dataset_name: str
    task: str
    predictions: dict = field(default_factory=dict)
    detection_table: pd.DataFrame | None = None
    mitigation_table: pd.DataFrame | None = None
    evolution_gad: pd.DataFrame | None = None
    evolution_mad: pd.DataFrame | None = None
    descriptor_weights: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


# =========================================================
# EVOLUCIÓN POR BLOQUES
# =========================================================

def metrics_for_block(
    y_true_all,
    y_pred_all,
    sensitive_all,
    end,
    thresholds,
    weights,
    class_aggregation,
):
    """
    Calcula métricas de rendimiento y equidad sobre el acumulado de individuos
    desde el inicio hasta `end` (exclusivo). Se emplea para la evolución por
    bloques durante el entrenamiento.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_b = y_true_all[:end]
    yp_b = y_pred_all[:end]
    s_b = sensitive_all[:end]

    acc = accuracy_score(y_b, yp_b)
    f1_mac = f1_score(y_b, yp_b, average="macro", zero_division=0)
    f1_wei = f1_score(y_b, yp_b, average="weighted", zero_division=0)

    fc_df = reporting.flatten_group_class_report(
        reporting.compute_group_class_fairness(
            y_b, yp_b, s_b, thresholds=thresholds, weights=weights,
        )
    )

    di_df = reporting.compute_normal_di_df(y_b, yp_b, s_b)

    if len(fc_df) > 0:
        mean_spd = fc_df["spd"].abs().mean()
        mean_eod = fc_df["eod"].abs().mean()
    else:
        mean_spd = mean_eod = float("nan")

    if len(di_df) > 0 and di_df["di"].notna().any():
        mean_di = float(np.nanmean(di_df["di"].to_numpy(dtype=float)))
    else:
        mean_di = float("nan")

    fg_df = pd.DataFrame(
        reporting.compute_group_fairness(
            y_b, yp_b, s_b, thresholds=thresholds, weights=weights,
            class_aggregation=class_aggregation,
        )
    ).T
    mean_disp = fg_df["disparity"].astype(float).mean() if len(fg_df) > 0 else float("nan")

    return {
        "n_acumulado": end,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wei,
        "mean_abs_spd": mean_spd,
        "mean_abs_eod": mean_eod,
        "mean_di": mean_di,
        "mean_disparity": mean_disp,
    }


def build_block_evolution(
    y_true_all,
    y_pred_all,
    sensitive_all,
    block_summaries,
    thresholds,
    weights,
    class_aggregation,
):
    """
    Construye la evolución de métricas acumuladas bloque a bloque sobre el
    conjunto proporcionado (siempre entrenamiento).
    """
    rows = []
    for summary in block_summaries:
        m = metrics_for_block(
            y_true_all, y_pred_all, sensitive_all,
            end=summary["end"],
            thresholds=thresholds, weights=weights,
            class_aggregation=class_aggregation,
        )
        m["bloque"] = summary["block_index"]
        rows.append(m)
    return pd.DataFrame(rows).set_index("bloque")


# =========================================================
# EXCLUSIÓN DEL ATRIBUTO SENSIBLE DE LA MATRIZ DE DESCRIPTORES
# =========================================================

def sensitive_feature_indices(
    feature_names: list,
    dataset_config,
    sensitive_mode: str,
) -> list[int]:
    """
    Localiza en la matriz de descriptores las columnas que codifican el atributo
    sensible, a partir de los patrones declarados en la configuración del
    conjunto.

    Se emplea en el análisis de ablación que compara la ejecución conservando el
    atributo sensible entre los descriptores frente a la que lo excluye. La
    correspondencia entre el modo de atributo sensible y los nombres de columna
    se declara de forma explícita por conjunto, ya que la codificación difiere
    entre loaders y una coincidencia por similitud de nombre podría retirar
    variables distintas de la deseada.

    Parámetros
    ----------
    feature_names : list
        Nombres de los descriptores devueltos por el loader.
    dataset_config : DatasetConfig
        Configuración del conjunto, que declara los patrones aplicables.
    sensitive_mode : str
        Modo de atributo sensible en uso.

    Retorna
    -------
    list[int]
        Índices de las columnas que codifican el atributo sensible. Lista vacía
        si el conjunto no declara patrones para ese modo.
    """
    patterns = getattr(dataset_config, "sensitive_feature_patterns", {}) or {}
    patterns = patterns.get(sensitive_mode, ())

    if not patterns:
        return []

    return [
        i for i, name in enumerate(feature_names)
        if any(str(name) == p or str(name).startswith(p) for p in patterns)
    ]


# =========================================================
# CARGA Y PARTICIÓN
# =========================================================

def load_and_split(
    dataset_name: str,
    sensitive_mode: str | None = None,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
    file_path: str | None = None,
    drop_sensitive_from_X: bool = False,
):
    """
    Carga un dataset mediante su loader y lo divide en train/test.

    La normalización NO se aplica aquí: la realiza internamente el clasificador
    sobre el conjunto de entrenamiento, evitando fuga de información.

    Parámetros
    ----------
    dataset_name : str
        Identificador del dataset (ver config.DATASETS).
    sensitive_mode : str | None, default=None
        Atributo sensible. Si es None, se usa el de la configuración.
    test_size : float
        Proporción de test.
    random_state : int
        Semilla de la partición.
    file_path : str | None
        Ruta al fichero. Si es None, el loader usa su ruta por defecto.

    Retorna
    -------
    tuple
        (X_train, X_test, y_train, y_test, s_train, s_test, feature_names, dataset_config)
    """
    dataset_config = cfg.get_dataset_config(dataset_name)
    mode = sensitive_mode if sensitive_mode is not None else dataset_config.sensitive_mode

    X, y, sensitive, feature_names = dataset_config.loader(
        file_path=file_path,
        sensitive_mode=mode,
        **dataset_config.loader_kwargs,
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    sensitive = np.asarray(sensitive)

    # Exclusión opcional del atributo sensible de la matriz de descriptores.
    # El vector de grupos sensibles se conserva intacto, ya que sigue siendo
    # necesario para evaluar la equidad y para construir la señal de disparidad.
    if drop_sensitive_from_X:
        idx = sensitive_feature_indices(feature_names, dataset_config, mode)
        if idx:
            excluidas = [feature_names[i] for i in idx]
            keep = [i for i in range(X.shape[1]) if i not in set(idx)]
            X = X[:, keep]
            feature_names = [feature_names[i] for i in keep]
            print(f"[{dataset_name}] descriptores excluidos ({len(excluidas)}): {excluidas}")
        else:
            print(
                f"[{dataset_name}] AVISO: se solicitó excluir el atributo sensible "
                f"'{mode}' de X, pero la configuración no declara patrones de "
                f"columna para ese modo. La matriz se mantiene sin cambios."
            )

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, s_train, s_test, feature_names, dataset_config


# =========================================================
# ENTRENAMIENTO DEL MODELO CLÁSICO (compartido)
# =========================================================

def _fit_classic_and_split(
    dataset_name: str,
    sensitive_mode: str | None,
    alpha: float,
    operator: str,
    test_size: float,
    random_state: int,
    file_path: str | None,
    metric_profile_kind: str | None,
    drop_sensitive_from_X: bool = False,
):
    """
    Carga, parte y entrena el LAMDA clásico. Devuelve todo lo necesario tanto
    para la detección (solo clásico) como para la mitigación (que reutiliza el
    split y el perfil). Centraliza lo común para no duplicar carga ni partición.

    Retorna
    -------
    dict
        Diccionario con split, clásico ajustado, predicción de test del clásico,
        perfil de métricas, umbrales y configuración del dataset.
    """
    (X_train, X_test, y_train, y_test,
     s_train, s_test, feature_names, dataset_config) = load_and_split(
        dataset_name=dataset_name,
        sensitive_mode=sensitive_mode,
        test_size=test_size,
        random_state=random_state,
        file_path=file_path,
        drop_sensitive_from_X=drop_sensitive_from_X,
    )

    n_classes = len(np.unique(y_train))
    profile = build_metric_profile(n_classes=n_classes, kind=metric_profile_kind)

    lamda_classic = LamdaClassifier(alpha=alpha, operator=operator)
    lamda_classic.fit(X_train, y_train)
    y_pred_classic = lamda_classic.predict(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "s_train": s_train, "s_test": s_test,
        "feature_names": feature_names,
        "dataset_config": dataset_config,
        "n_classes": int(n_classes),
        "profile": profile,
        "classic": lamda_classic,
        "y_pred_classic": y_pred_classic,
    }


# =========================================================
# FASE DE DETECCIÓN (solo LAMDA clásico)
# =========================================================

def run_detection(
    dataset_name: str,
    sensitive_mode: str | None = None,
    alpha: float = cfg.ALPHA,
    operator: str = cfg.AGGREGATION_OPERATOR,
    thresholds: FairnessThresholds | None = None,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
    file_path: str | None = None,
    metric_profile_kind: str | None = None,
    weights: FairnessWeights | None = None,
    class_aggregation: str | None = None,
    drop_sensitive_from_X: bool = False,
) -> ExperimentResult:
    """
    Ejecuta la fase de DETECCIÓN para un dataset: caracteriza la disparidad de
    partida del LAMDA clásico sobre el conjunto de test, sin aplicar mitigación.

    Esta es la fase que se compara con la literatura. No
    interviene Fair-GAD ni Fair-MAD. Las métricas se adaptan al tipo de tarea:
    SPD, EOD y DI en binario; SPD multiclase y MEO en multiclase.

    Retorna
    -------
    ExperimentResult
        Con detection_table poblada; el resto de campos quedan vacíos.
    """
    base = _fit_classic_and_split(
        dataset_name=dataset_name,
        sensitive_mode=sensitive_mode,
        alpha=alpha,
        operator=operator,
        test_size=test_size,
        random_state=random_state,
        file_path=file_path,
        metric_profile_kind=metric_profile_kind,
        drop_sensitive_from_X=drop_sensitive_from_X,
    )

    profile = base["profile"]
    task = profile.kind
    # La composición de D_g (qué métricas intervienen y con qué peso) y la
    # agregación sobre clases pueden fijarse explícitamente desde el notebook.
    # Si no se indican, se toman del perfil derivado del tipo de tarea.
    weights = profile.weights if weights is None else weights
    class_aggregation = (
        profile.class_aggregation if class_aggregation is None else class_aggregation
    )

    if thresholds is None:
        thresholds = cfg.FAIRNESS_THRESHOLDS

    detection_row = reporting.summarize_model_metrics(
        base["y_test"], base["y_pred_classic"], base["s_test"], task=task,
        thresholds=thresholds, weights=weights,
        class_aggregation=class_aggregation,
    )
    detection_row = {"Modelo": "LAMDA clásico", **detection_row}
    detection_table = pd.DataFrame([detection_row]).set_index("Modelo")

    metadata = {
        "dataset": dataset_name,
        "fase": "deteccion",
        "sensitive_mode": sensitive_mode or base["dataset_config"].sensitive_mode,
        "task": task,
        "n_classes": base["n_classes"],
        "alpha": alpha,
        "operator": operator,
        "test_size": test_size,
        "random_state": random_state,
        "w_spd": weights.spd,
        "w_eod": weights.eod,
        "w_di": weights.di,
        "normalize_disparity": weights.normalize,
        "class_aggregation": class_aggregation,
        "thr_spd": thresholds.spd,
        "thr_eod": thresholds.eod,
        "thr_di": thresholds.di,
        "drop_sensitive_from_X": drop_sensitive_from_X,
        "n_features": int(base["X_train"].shape[1]),
    }

    return ExperimentResult(
        dataset_name=dataset_name,
        task=task,
        predictions={
            "y_test": base["y_test"],
            "sensitive_test": base["s_test"],
            "classic": base["y_pred_classic"],
        },
        detection_table=detection_table,
        metadata=metadata,
    )


# =========================================================
# FASE DE MITIGACIÓN (tres modelos)
# =========================================================

def run_mitigation(
    dataset_name: str,
    sensitive_mode: str | None = None,
    alpha: float = cfg.ALPHA,
    operator: str = cfg.AGGREGATION_OPERATOR,
    thresholds: FairnessThresholds | None = None,
    lambda_fair: float = cfg.LAMBDA_FAIR,
    eta_fair: float = cfg.ETA_FAIR,
    tau_j: float = cfg.TAU_J,
    update_every_n_blocks: int = cfg.UPDATE_EVERY_N_BLOCKS,
    block_size: int = cfg.BLOCK_SIZE,
    min_group_size: int = cfg.MIN_GROUP_SIZE,
    min_positive_size: int = cfg.MIN_POSITIVE_SIZE,
    default_group_disparity: float = cfg.DEFAULT_GROUP_DISPARITY,
    use_cumulative_history: bool = cfg.USE_CUMULATIVE_HISTORY,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
    file_path: str | None = None,
    metric_profile_kind: str | None = None,
    weights: FairnessWeights | None = None,
    class_aggregation: str | None = None,
    drop_sensitive_from_X: bool = False,
    disparity_mode_gad: str = "directional",
    disparity_mode_mad: str = "symmetric",
) -> ExperimentResult:
    """
    Ejecuta la fase de MITIGACIÓN para un dataset: entrena Fair-GAD y Fair-MAD
    además del clásico y compara los tres sobre el conjunto de test con los
    mecanismos de equidad congelados.

    Produce también la evolución por bloques durante el entrenamiento y los
    pesos de descriptores de Fair-MAD.

    Todos los parámetros del experimento se reciben como argumentos, de modo que
    el notebook controla la configuración.

    Retorna
    -------
    ExperimentResult
        Con mitigation_table, evolution_gad, evolution_mad y descriptor_weights
        poblados.
    """
    base = _fit_classic_and_split(
        dataset_name=dataset_name,
        sensitive_mode=sensitive_mode,
        alpha=alpha,
        operator=operator,
        test_size=test_size,
        random_state=random_state,
        file_path=file_path,
        metric_profile_kind=metric_profile_kind,
        drop_sensitive_from_X=drop_sensitive_from_X,
    )

    X_train, X_test = base["X_train"], base["X_test"]
    y_train, y_test = base["y_train"], base["y_test"]
    s_train, s_test = base["s_train"], base["s_test"]
    feature_names = base["feature_names"]
    dataset_config = base["dataset_config"]

    profile = base["profile"]
    task = profile.kind
    # Composición de D_g y agregación sobre clases: configurables desde el
    # notebook. Gobiernan tanto el reporte como la señal de penalización de
    # Fair-GAD y Fair-MAD, por lo que deben fijarse antes de ejecutar y
    # documentarse junto con los resultados.
    weights = profile.weights if weights is None else weights
    class_aggregation = (
        profile.class_aggregation if class_aggregation is None else class_aggregation
    )

    if thresholds is None:
        thresholds = cfg.FAIRNESS_THRESHOLDS

    y_pred_classic = base["y_pred_classic"]

    # --- Fair-GAD ---
    lamda_fair_gad = LamdaFairGADClassifier(
        alpha=alpha,
        operator=operator,
        lambda_=lambda_fair,
        block_size=block_size,
        fairness_thresholds=thresholds,
        fairness_weights=weights,
        clip=True,
        min_group_size=min_group_size,
        min_positive_size=min_positive_size,
        default_group_disparity=default_group_disparity,
        class_aggregation=class_aggregation,
        use_cumulative_history=use_cumulative_history,
        disparity_mode=disparity_mode_gad,
    )
    lamda_fair_gad.fit(X_train, y_train)
    result_gad_train = lamda_fair_gad.predict_blocks(
        X=X_train, y_true=y_train, sensitive=s_train, return_details=True,
    )
    y_pred_fair_gad = lamda_fair_gad.predict_fair_from_disparities(
        X=X_test, sensitive=s_test,
    )

    # --- Fair-MAD ---
    lamda_fair_mad = LamdaFairMADClassifier(
        alpha=alpha,
        operator=operator,
        eta=eta_fair,
        tau_j=tau_j,
        block_size=block_size,
        update_every_n_blocks=update_every_n_blocks,
        fairness_thresholds=thresholds,
        fairness_weights=weights,
        clip=True,
        min_group_size=min_group_size,
        min_positive_size=min_positive_size,
        default_group_disparity=default_group_disparity,
        class_aggregation=class_aggregation,
        use_cumulative_history=use_cumulative_history,
        disparity_mode=disparity_mode_mad,
    )
    lamda_fair_mad.fit(X_train, y_train)
    result_mad_train = lamda_fair_mad.predict_blocks(
        X=X_train, y_true=y_train, sensitive=s_train, return_details=True,
    )
    y_pred_fair_mad = lamda_fair_mad.predict_fair_from_weights(X=X_test)

    # --- Tabla de mitigación (tres modelos sobre test) ---
    mitigation_rows = []
    for name, y_pred in [
        ("LAMDA clásico", y_pred_classic),
        ("Fair-GAD", y_pred_fair_gad),
        ("Fair-MAD", y_pred_fair_mad),
    ]:
        r = reporting.summarize_model_metrics(
            y_test, y_pred, s_test, task=task,
            thresholds=thresholds, weights=weights,
            class_aggregation=class_aggregation,
        )
        mitigation_rows.append({"Modelo": name, **r})
    mitigation_table = pd.DataFrame(mitigation_rows).set_index("Modelo")

    # --- Evolución por bloques durante el entrenamiento ---
    evolution_gad = build_block_evolution(
        y_train, result_gad_train["y_pred_final"], s_train,
        result_gad_train["block_summaries"],
        thresholds=thresholds, weights=weights, class_aggregation=class_aggregation,
    )
    evolution_mad = build_block_evolution(
        y_train, result_mad_train["y_pred_final"], s_train,
        result_mad_train["block_summaries"],
        thresholds=thresholds, weights=weights, class_aggregation=class_aggregation,
    )

    # --- Pesos de descriptores de Fair-MAD ---
    descriptor_weights = pd.DataFrame({
        "feature": feature_names,
        "w_j": lamda_fair_mad.descriptor_weights_,
        "D_j": lamda_fair_mad.descriptor_disparities_,
    }).sort_values("w_j")

    metadata = {
        "dataset": dataset_name,
        "fase": "mitigacion",
        "sensitive_mode": sensitive_mode or dataset_config.sensitive_mode,
        "task": task,
        "n_classes": base["n_classes"],
        "alpha": alpha,
        "operator": operator,
        "lambda_fair": lambda_fair,
        "eta_fair": eta_fair,
        "block_size": block_size,
        "update_every_n_blocks": update_every_n_blocks,
        "class_aggregation": class_aggregation,
        "w_spd": weights.spd,
        "w_eod": weights.eod,
        "w_di": weights.di,
        "normalize_disparity": weights.normalize,
        "thr_spd": thresholds.spd,
        "thr_eod": thresholds.eod,
        "thr_di": thresholds.di,
        "tau_j": tau_j,
        "disparity_mode_gad": disparity_mode_gad,
        "disparity_mode_mad": disparity_mode_mad,
        "drop_sensitive_from_X": drop_sensitive_from_X,
        "n_features": int(X_train.shape[1]),
        "profile_kind": profile.kind,
        "test_size": test_size,
        "random_state": random_state,
    }

    return ExperimentResult(
        dataset_name=dataset_name,
        task=task,
        predictions={
            "y_test": y_test,
            "sensitive_test": s_test,
            "classic": y_pred_classic,
            "fair_gad": y_pred_fair_gad,
            "fair_mad": y_pred_fair_mad,
        },
        mitigation_table=mitigation_table,
        evolution_gad=evolution_gad,
        evolution_mad=evolution_mad,
        descriptor_weights=descriptor_weights,
        metadata=metadata,
    )


# =========================================================
# PERSISTENCIA
# =========================================================

def persist_result(
    result: ExperimentResult,
    results_dir: str = cfg.RESULTS_DIR,
) -> dict[str, str]:
    """
    Persiste las tablas de un experimento en disco, para que el notebook de
    agregación pueda leerlas.

    Escribe, por dataset:
    - {dataset}_deteccion.csv
    - {dataset}_mitigacion.csv
    - {dataset}_evolucion_gad.csv
    - {dataset}_evolucion_mad.csv
    - {dataset}_descriptores_mad.csv
    - {dataset}_metadata.csv

    Parámetros
    ----------
    result : ExperimentResult
        Resultado a persistir.
    results_dir : str
        Directorio de salida (se crea si no existe).

    Retorna
    -------
    dict[str, str]
        Rutas de los ficheros escritos.
    """
    os.makedirs(results_dir, exist_ok=True)
    name = result.dataset_name
    paths = {}

    def _save(df, suffix, index=True):
        path = os.path.join(results_dir, f"{name}_{suffix}.csv")
        df.to_csv(path, index=index)
        paths[suffix] = path

    if result.detection_table is not None:
        _save(result.detection_table, "deteccion")
    if result.mitigation_table is not None:
        _save(result.mitigation_table, "mitigacion")
    if result.evolution_gad is not None:
        _save(result.evolution_gad, "evolucion_gad")
    if result.evolution_mad is not None:
        _save(result.evolution_mad, "evolucion_mad")
    if result.descriptor_weights is not None:
        _save(result.descriptor_weights, "descriptores_mad", index=False)

    meta_df = pd.DataFrame([result.metadata])
    _save(meta_df, "metadata", index=False)

    return paths
