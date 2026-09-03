"""
Construcción de métricas e informes de los experimentos.

Este módulo reúne las funciones de evaluación de rendimiento y equidad usadas en
los experimentos, extraídas del flujo de trabajo validado en notebooks previos y
organizadas para su reutilización. Ninguna función fija parámetros del
experimento: todos (umbrales, pesos, agregación) se reciben como argumentos, de
modo que el notebook mantiene el control completo de la configuración.

Contenido
---------
- Métricas de rendimiento predictivo (accuracy, F1).
- Informes de equidad por grupo y por (grupo, clase), delegando en el módulo
  de disparidad.
- Resúmenes de disparidad agregada (mean/max disparity, |SPD|, |EOD|, DI
  simétrico).
- DI normal (no simétrico) con fines de reporte.
- Métricas multiclase de reporte comparativo (SPD multiclase y MEO), separadas
  de la señal de penalización y usadas solo en las tablas de detección.
- Construcción de la fila de la tabla final por modelo, adaptada al tipo de
  tarea (binaria o multiclase).

Las funciones que dependen de umbrales o pesos los reciben explícitamente. Se
proporcionan valores por defecto neutros para poder invocarlas de forma directa,
pero el notebook debe pasar la configuración deseada.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from lamda.fairness.disparity import (
    FairnessThresholds,
    FairnessWeights,
    group_disparity_report,
    group_class_disparity_report,
)
from lamda.fairness.metrics import (
    disparate_impact,
    multiclass_fairness_report,
)


# =========================================================
# RENDIMIENTO PREDICTIVO
# =========================================================

def evaluate_performance(y_true, y_pred, model_name: str) -> dict:
    """
    Calcula métricas de rendimiento predictivo para un modelo.

    En problemas binarios se añade F1_binary. En problemas multiclase se deja
    como NaN para mantener una tabla común entre datasets.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    model_name : str
        Nombre del modelo (para etiquetar la fila).

    Retorna
    -------
    dict
        Métricas de rendimiento.
    """
    unique_classes = np.unique(y_true)
    f1_binary = np.nan

    if len(unique_classes) == 2:
        f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)

    return {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_binary": f1_binary,
    }


# =========================================================
# INFORMES DE EQUIDAD (DELEGAN EN disparity.py)
# =========================================================

def compute_group_fairness(
    y_true,
    y_pred,
    sensitive_values,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    class_aggregation: str = "mean",
) -> dict:
    """
    Calcula el informe agregado de fairness por grupo sensible (D_g por grupo).

    Los umbrales, pesos y la agregación se reciben como argumentos para que el
    notebook controle el perfil de métricas aplicado.
    """
    return group_disparity_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive_values,
        thresholds=thresholds,
        weights=weights,
        class_aggregation=class_aggregation,
    )


def compute_group_class_fairness(
    y_true,
    y_pred,
    sensitive_values,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
) -> dict:
    """
    Calcula el informe detallado grupo-clase en esquema one-vs-rest.

    Nota: group_class_disparity_report no agrega sobre clases, por lo que no
    recibe class_aggregation.
    """
    return group_class_disparity_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive_values,
        thresholds=thresholds,
        weights=weights,
    )


def flatten_group_class_report(report: dict) -> pd.DataFrame:
    """
    Convierte un informe anidado grupo-clase en un DataFrame plano.

    Parámetros
    ----------
    report : dict
        Informe anidado {grupo: {clase: {metricas}}}.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas group, target_class y las métricas.
    """
    rows = []

    for group, class_dict in report.items():
        for target_class, metrics in class_dict.items():
            rows.append(
                {
                    "group": group,
                    "target_class": target_class,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def compute_normal_di_df(y_true, y_pred, sensitive_values) -> pd.DataFrame:
    """
    Calcula el Disparate Impact NORMAL (no simétrico) para cada par
    (grupo, clase) bajo el esquema one-vs-rest, tratando cada clase como
    resultado favorable.

    A diferencia del DI simétrico empleado en D_g (ideal 1, valores en [0, 1]),
    el DI normal tiene ideal 1 y puede superar 1 cuando el grupo recibe la clase
    favorable con más frecuencia que el resto. Se usa solo para evaluación e
    interpretación. Los casos no evaluables (denominador nulo) se marcan como
    NaN para que no contaminen las agregaciones (se omiten con nanmean/nanmin).

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive_values : np.ndarray
        Vector de grupos sensibles.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas group, target_class y di.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_values = np.asarray(sensitive_values)

    groups = np.unique(sensitive_values)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    rows = []
    for group in groups:
        for target_class in classes:
            di = disparate_impact(
                y_pred=y_pred,
                sensitive=sensitive_values,
                group=group,
                positive_label=target_class,
                default_value=np.nan,  # no evaluable -> NaN
            )
            rows.append(
                {
                    "group": group,
                    "target_class": target_class,
                    "di": di,
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# REPORTE MULTICLASE COMPARATIVO (SPD-mc y MEO)
# =========================================================

def compute_multiclass_report_df(
    y_true,
    y_pred,
    sensitive_values,
    spd_aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Construye el reporte multiclase comparativo (SPD multiclase y MEO) por grupo.

    Estas métricas siguen la convención de la literatura de equidad multiclase
    (Alghamdi et al., 2022) y se usan exclusivamente en las tablas de detección
    de los datasets multiclase, para comparar con la literatura. No intervienen
    en el cálculo de D_g.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive_values : np.ndarray
        Vector de grupos sensibles.
    spd_aggregation : str, default="mean"
        Agregación del SPD multiclase ("mean" o "max").

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas group, spd_multiclass y meo.
    """
    report = multiclass_fairness_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive_values,
        spd_aggregation=spd_aggregation,
    )

    rows = [
        {"group": group, "spd_multiclass": vals["spd_multiclass"], "meo": vals["meo"]}
        for group, vals in report.items()
    ]

    return pd.DataFrame(rows)


# =========================================================
# MÉTRICAS AGREGADAS POR MODELO (adaptadas al tipo de tarea)
# =========================================================

def summarize_model_metrics(
    y_true,
    y_pred,
    sensitive_values,
    task: str,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    class_aggregation: str = "mean",
    spd_aggregation: str = "mean",
) -> dict:
    """
    Resume rendimiento y equidad de un modelo en una única fila, adaptando las
    métricas de equidad al tipo de tarea.

    Para tareas binarias se reportan las métricas one-vs-rest habituales
    (|SPD|, |EOD| y DI normal). Para tareas multiclase se añaden
    las métricas de reporte comparativo (SPD multiclase y MEO) y se omite DI,
    coherentemente con el perfil multiclase.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive_values : np.ndarray
        Vector de grupos sensibles.
    task : {"binary", "multiclass"}
        Tipo de tarea.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación de D_g.
    class_aggregation : str, default="mean"
        Agregación sobre clases para D_g.
    spd_aggregation : str, default="mean"
        Agregación del SPD multiclase (solo tareas multiclase).

    Retorna
    -------
    dict
        Fila con métricas de rendimiento y equidad.
    """
    # Rendimiento.
    perf = evaluate_performance(y_true, y_pred, model_name="")

    # Disparidad agregada D_g por grupo.
    fg_report = compute_group_fairness(
        y_true, y_pred, sensitive_values,
        thresholds=thresholds, weights=weights,
        class_aggregation=class_aggregation,
    )
    fg_df = pd.DataFrame(fg_report).T
    mean_disp = fg_df["disparity"].astype(float).mean() if len(fg_df) > 0 else float("nan")
    max_disp = fg_df["disparity"].astype(float).max() if len(fg_df) > 0 else float("nan")

    # Métricas detalladas grupo-clase (one-vs-rest).
    fc_df = flatten_group_class_report(
        compute_group_class_fairness(
            y_true, y_pred, sensitive_values,
            thresholds=thresholds, weights=weights,
        )
    )

    if len(fc_df) > 0:
        mean_abs_spd = fc_df["spd"].abs().mean()
        max_abs_spd = fc_df["spd"].abs().max()
        mean_abs_eod = fc_df["eod"].abs().mean()
        max_abs_eod = fc_df["eod"].abs().max()
    else:
        mean_abs_spd = max_abs_spd = mean_abs_eod = max_abs_eod = float("nan")

    # Impacto dispar en su versión NORMAL (cociente de tasas, ideal 1). Es la
    # versión que se reporta y con la que se contrasta con la literatura. La
    # versión simétrica interviene únicamente en la construcción de la
    # disparidad agregada D_g, donde se requiere una magnitud acotada y
    # monótona; no se reporta para evitar confundir dos escalas distintas.
    # Los pares (grupo, clase) no evaluables quedan como NaN y se omiten de la
    # agregación mediante nanmean y nanmin.
    di_df = compute_normal_di_df(y_true, y_pred, sensitive_values)
    if len(di_df) > 0 and di_df["di"].notna().any():
        di_values = di_df["di"].to_numpy(dtype=float)
        mean_di = float(np.nanmean(di_values))
        min_di = float(np.nanmin(di_values))
    else:
        mean_di = min_di = float("nan")

    row = {
        "n": len(y_true),
        "Accuracy": perf["Accuracy"],
        "F1_macro": perf["F1_macro"],
        "F1_weighted": perf["F1_weighted"],
        "F1_binary": perf["F1_binary"],
        "mean_disparity": mean_disp,
        "max_disparity": max_disp,
        "mean_abs_spd": mean_abs_spd,
        "max_abs_spd": max_abs_spd,
        "mean_abs_eod": mean_abs_eod,
        "max_abs_eod": max_abs_eod,
        "mean_di": mean_di,
        "min_di": min_di,
    }

    # En multiclase se añaden las métricas de reporte comparativo.
    if task == "multiclass":
        mc_df = compute_multiclass_report_df(
            y_true, y_pred, sensitive_values,
            spd_aggregation=spd_aggregation,
        )
        if len(mc_df) > 0:
            row["mean_spd_multiclass"] = mc_df["spd_multiclass"].mean()
            row["max_spd_multiclass"] = mc_df["spd_multiclass"].max()
            row["mean_meo"] = mc_df["meo"].mean()
            row["max_meo"] = mc_df["meo"].max()
        else:
            row["mean_spd_multiclass"] = row["max_spd_multiclass"] = float("nan")
            row["mean_meo"] = row["max_meo"] = float("nan")

    return row
