"""
Cálculo de disparidad agregada por grupo sensible.

Este módulo implementa una medida escalar de disparidad D_g a partir de
métricas de equidad calculadas para cada grupo sensible.

El cálculo es compatible con clasificación binaria y multiclase.

Formulación
-----------
Sea g un grupo sensible y sea C el conjunto de clases.

Para cada clase c ∈ C se construye un problema one-vs-rest, considerando
la clase c como resultado favorable. Sobre esta base se calculan:

    SPD_g^(c)     = P(Y_hat = c | S = g) - P(Y_hat = c | S != g)
    EOD_g^(c)     = P(Y_hat = c | Y = c, S = g) - P(Y_hat = c | Y = c, S != g)
    DI_sym_g^(c)  = min(DI_g^(c), 1 / DI_g^(c))

donde:

    DI_g^(c) = P(Y_hat = c | S = g) / P(Y_hat = c | S != g)

A partir de estas métricas se define la disparidad por grupo y clase:

    D_g^(c) =
        w_spd * max(0, |SPD_g^(c)| - tau_spd) +
        w_eod * max(0, |EOD_g^(c)| - tau_eod) +
        w_di  * max(0, tau_di - DI_sym_g^(c))

Finalmente, la disparidad agregada del grupo se obtiene como:

    D_g = (1 / |C|) * sum_{c in C} D_g^(c)

Interpretación
--------------
- Si una métrica cumple el umbral, su contribución es 0.
- Si lo viola, contribuye positivamente a la disparidad del grupo.
- D_g = 0 indica ausencia de violación respecto a los umbrales definidos.
- Cuanto mayor es D_g, mayor es la disparidad asociada al grupo.

Flujo
-----
1. Se identifican las clases presentes en y_true e y_pred.
2. Para cada grupo sensible y para cada clase:
   - se calculan SPD, EOD y DI simétrico en esquema one-vs-rest.
   - se calcula la disparidad D_g^(c).
3. Se agregan las disparidades de todas las clases.
4. Se devuelve el diccionario final {grupo: D_g}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lamda.fairness.metrics import (
    disparate_impact_symmetric,
    equal_opportunity_difference,
    statistical_parity_difference,
)


# =========================================================
# CONFIGURACIÓN
# =========================================================

@dataclass(frozen=True)
class FairnessThresholds:
    """
    Umbrales de equidad utilizados en el cálculo de disparidad.

    Parámetros
    ----------
    spd : float, default=0.1
        Umbral máximo aceptable para la magnitud de SPD.
    eod : float, default=0.1
        Umbral máximo aceptable para la magnitud de EOD.
    di : float, default=0.8
        Umbral mínimo aceptable para DI simétrico.
    """

    spd: float = 0.1
    eod: float = 0.1
    di: float = 0.8


@dataclass(frozen=True)
class FairnessWeights:
    """
    Pesos de agregación utilizados en la disparidad por grupo.

    Parámetros
    ----------
    spd : float, default=1.0
        Peso de la componente SPD.
    eod : float, default=1.0
        Peso de la componente EOD.
    di : float, default=1.0
        Peso de la componente DI simétrico.
    """

    spd: float = 1.0
    eod: float = 1.0
    di: float = 1.0
    normalize: bool = False


def active_weight_sum(weights: "FairnessWeights") -> float:
    """
    Devuelve la suma de los pesos de las componentes de la disparidad.

    Se emplea para normalizar D_g cuando `weights.normalize` es True, de modo
    que la disparidad quede acotada en la misma escala con independencia de
    cuántas métricas compongan la señal. Sin normalizar, una composición de
    tres métricas produce valores sistemáticamente mayores que una de dos, lo
    que altera la intensidad efectiva de la penalización (lambda o eta) y hace
    no comparables las disparidades entre configuraciones o entre tareas.
    """
    return float(weights.spd + weights.eod + weights.di)


# =========================================================
# VALIDACIÓN
# =========================================================

def validate_fairness_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Valida las entradas necesarias para el cálculo de disparidades.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Entradas validadas y convertidas a arrays unidimensionales.

    Lanza
    ------
    ValueError
        Si las longitudes no coinciden.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(sensitive)):
        raise ValueError("y_true, y_pred y sensitive deben tener la misma longitud.")

    return y_true, y_pred, sensitive


def validate_thresholds(thresholds: FairnessThresholds | None) -> FairnessThresholds:
    """
    Valida o crea la configuración de umbrales.

    Parámetros
    ----------
    thresholds : FairnessThresholds | None
        Umbrales de equidad.

    Retorna
    -------
    FairnessThresholds
        Umbrales validados.
    """
    if thresholds is None:
        thresholds = FairnessThresholds()

    if thresholds.spd < 0 or thresholds.eod < 0 or thresholds.di < 0:
        raise ValueError("Los umbrales deben ser mayores o iguales que 0.")

    return thresholds


def validate_weights(weights: FairnessWeights | None) -> FairnessWeights:
    """
    Valida o crea la configuración de pesos.

    Parámetros
    ----------
    weights : FairnessWeights | None
        Pesos de agregación.

    Retorna
    -------
    FairnessWeights
        Pesos validados.
    """
    if weights is None:
        weights = FairnessWeights()

    if weights.spd < 0 or weights.eod < 0 or weights.di < 0:
        raise ValueError("Los pesos deben ser mayores o iguales que 0.")

    return weights


# =========================================================
# UTILIDADES
# =========================================================

def unique_sensitive_groups(sensitive: np.ndarray) -> np.ndarray:
    """
    Obtiene los grupos sensibles únicos presentes en los datos.

    Parámetros
    ----------
    sensitive : np.ndarray
        Vector de grupos sensibles.

    Retorna
    -------
    np.ndarray
        Array con los grupos sensibles únicos.
    """
    sensitive = np.asarray(sensitive).reshape(-1)
    return np.unique(sensitive)


def unique_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Obtiene el conjunto de clases presentes en y_true e y_pred.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.

    Retorna
    -------
    np.ndarray
        Array con las clases únicas observadas.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.unique(np.concatenate([y_true, y_pred]))


def group_support(
    sensitive: np.ndarray,
    group: Any,
) -> int:
    """
    Calcula el tamaño muestral de un grupo sensible.

    Parámetros
    ----------
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo cuya frecuencia se desea calcular.

    Retorna
    -------
    int
        Número de individuos pertenecientes al grupo.
    """
    sensitive = np.asarray(sensitive).reshape(-1)
    return int(np.sum(sensitive == group))


def class_support(
    y_true: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    target_class: Any,
) -> int:
    """
    Calcula el número de ejemplos reales de una clase dentro de un grupo.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    target_class : Any
        Clase objetivo del esquema one-vs-rest.

    Retorna
    -------
    int
        Número de ejemplos con etiqueta real igual a target_class en el grupo.
    """
    y_true = np.asarray(y_true).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)

    mask = sensitive == group
    return int(np.sum(y_true[mask] == target_class))


# =========================================================
# VIOLACIONES POR MÉTRICA
# =========================================================

def spd_violation(
    spd_value: float,
    tau_spd: float,
) -> float:
    """
    Calcula la violación del umbral de SPD.

    Parámetros
    ----------
    spd_value : float
        Valor de SPD.
    tau_spd : float
        Umbral de SPD.

    Retorna
    -------
    float
        Violación no negativa del umbral.
    """
    return max(0.0, abs(float(spd_value)) - float(tau_spd))


def eod_violation(
    eod_value: float,
    tau_eod: float,
) -> float:
    """
    Calcula la violación del umbral de EOD.

    Parámetros
    ----------
    eod_value : float
        Valor de EOD.
    tau_eod : float
        Umbral de EOD.

    Retorna
    -------
    float
        Violación no negativa del umbral.
    """
    return max(0.0, abs(float(eod_value)) - float(tau_eod))


def di_violation(
    di_sym_value: float,
    tau_di: float,
) -> float:
    """
    Calcula la violación del umbral de DI simétrico.

    Parámetros
    ----------
    di_sym_value : float
        Valor de DI simétrico.
    tau_di : float
        Umbral mínimo aceptable.

    Retorna
    -------
    float
        Violación no negativa del umbral.
    """
    return max(0.0, float(tau_di) - float(di_sym_value))


# =========================================================
# MÉTRICAS POR GRUPO Y CLASE
# =========================================================

def group_class_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    target_class: Any,
) -> dict[str, float]:
    """
    Calcula métricas de fairness para un grupo y una clase concreta
    bajo un esquema one-vs-rest.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    target_class : Any
        Clase tratada como favorable.

    Retorna
    -------
    dict[str, float]
        Diccionario con SPD, EOD y DI simétrico para el par (grupo, clase).
    """
    spd = statistical_parity_difference(
        y_pred=y_pred,
        sensitive=sensitive,
        group=group,
        positive_label=target_class,
    )

    eod = equal_opportunity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        group=group,
        positive_label=target_class,
    )

    di_sym = disparate_impact_symmetric(
        y_pred=y_pred,
        sensitive=sensitive,
        group=group,
        positive_label=target_class,
    )

    return {
        "spd": float(spd),
        "eod": float(eod),
        "di_sym": float(di_sym),
    }


def group_class_disparity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    target_class: Any,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
) -> float:
    """
    Calcula la disparidad D_g^(c) para un grupo y una clase concreta.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    target_class : Any
        Clase tratada como favorable.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.

    Retorna
    -------
    float
        Disparidad del par (grupo, clase).
    """
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    metrics = group_class_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        group=group,
        target_class=target_class,
    )

    spd_term = weights.spd * spd_violation(metrics["spd"], thresholds.spd)
    eod_term = weights.eod * eod_violation(metrics["eod"], thresholds.eod)
    di_term = weights.di * di_violation(metrics["di_sym"], thresholds.di)

    disparity = spd_term + eod_term + di_term

    # Normalización opcional por la suma de pesos activos. Desactivada por
    # defecto para preservar el comportamiento previo del método.
    if weights.normalize:
        total = active_weight_sum(weights)
        disparity = disparity / total if total > 0 else 0.0

    return float(disparity)


# =========================================================
# DISPARIDAD AGREGADA POR GRUPO
# =========================================================

def group_disparity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    class_aggregation: str = "mean",
) -> float:
    """
    Calcula la disparidad agregada D_g para un grupo sensible.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases. Opciones:
        - "mean": media de D_g^(c)
        - "max": máximo de D_g^(c)

    Retorna
    -------
    float
        Disparidad agregada del grupo.
    """
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    classes = unique_classes(y_true, y_pred)

    class_disparities = [
        group_class_disparity(
            y_true=y_true,
            y_pred=y_pred,
            sensitive=sensitive,
            group=group,
            target_class=target_class,
            thresholds=thresholds,
            weights=weights,
        )
        for target_class in classes
    ]

    if len(class_disparities) == 0:
        return 0.0

    if class_aggregation == "mean":
        return float(np.mean(class_disparities))

    if class_aggregation == "max":
        return float(np.max(class_disparities))

    raise ValueError("class_aggregation debe ser 'mean' o 'max'.")


def all_group_disparities(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    min_group_size: int = 1,
    min_positive_size: int = 1,
    default_for_small_groups: float = 0.0,
    class_aggregation: str = "mean",
) -> dict[Any, float]:
    """
    Calcula la disparidad agregada para todos los grupos sensibles presentes.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    min_group_size : int, default=1
        Soporte mínimo requerido para evaluar un grupo sensible.
        Si el grupo no alcanza este tamaño, se le asigna default_for_small_groups.
    min_positive_size : int, default=1
        Soporte mínimo requerido de ejemplos reales de una clase dentro del
        grupo para calcular EOD de forma estable. Si el par (grupo, clase)
        no alcanza este soporte, esa clase se excluye del cálculo de D_g
        en lugar de anular el grupo completo.
    default_for_small_groups : float, default=0.0
        Valor asignado a grupos que no alcanzan min_group_size.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.

    Retorna
    -------
    dict[Any, float]
        Diccionario {grupo: D_g}.
    """
    y_true, y_pred, sensitive = validate_fairness_inputs(y_true, y_pred, sensitive)
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    disparities: dict[Any, float] = {}
    classes = unique_classes(y_true, y_pred)

    for group in unique_sensitive_groups(sensitive):
        n_group = group_support(sensitive, group)

        # Guard de tamaño mínimo a nivel de grupo: si el grupo es demasiado
        # pequeño para cualquier estimación fiable, se asigna el valor por defecto.
        if n_group < min_group_size:
            disparities[group] = float(default_for_small_groups)
            continue

        # Para cada clase se calcula D_g^(c) de forma independiente.
        # Si el par (grupo, clase) no tiene suficientes positivos reales para
        # calcular EOD de forma estable, esa clase se excluye de la agregación
        # en lugar de anular todo el grupo.
        class_disparities = []

        for target_class in classes:
            n_positive = class_support(
                y_true=y_true,
                sensitive=sensitive,
                group=group,
                target_class=target_class,
            )

            if n_positive < min_positive_size:
                # No hay suficientes positivos reales para esta clase en este grupo.
                # Se omite su contribución al agregado en lugar de penalizar el grupo.
                continue

            class_disparities.append(
                group_class_disparity(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive=sensitive,
                    group=group,
                    target_class=target_class,
                    thresholds=thresholds,
                    weights=weights,
                )
            )

        if len(class_disparities) == 0:
            # Ninguna clase tenía soporte suficiente: se asigna el valor por defecto.
            disparities[group] = float(default_for_small_groups)
            continue

        if class_aggregation == "mean":
            disparities[group] = float(np.mean(class_disparities))
        elif class_aggregation == "max":
            disparities[group] = float(np.max(class_disparities))
        else:
            raise ValueError("class_aggregation debe ser 'mean' o 'max'.")

    return disparities


# =========================================================
# INFORME DETALLADO
# =========================================================

def group_disparity_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    class_aggregation: str = "mean",
) -> dict[Any, dict[str, float]]:
    """
    Genera un informe agregado de fairness por grupo sensible.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.

    Retorna
    -------
    dict[Any, dict[str, float]]
        Diccionario con soporte y disparidad agregada por grupo.
    """
    y_true, y_pred, sensitive = validate_fairness_inputs(y_true, y_pred, sensitive)
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    report: dict[Any, dict[str, float]] = {}
    classes = unique_classes(y_true, y_pred)

    for group in unique_sensitive_groups(sensitive):
        per_class_disparities = [
            group_class_disparity(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
                group=group,
                target_class=target_class,
                thresholds=thresholds,
                weights=weights,
            )
            for target_class in classes
        ]

        if class_aggregation == "mean":
            disparity_value = float(np.mean(per_class_disparities)) if len(per_class_disparities) > 0 else 0.0
        elif class_aggregation == "max":
            disparity_value = float(np.max(per_class_disparities)) if len(per_class_disparities) > 0 else 0.0
        else:
            raise ValueError("class_aggregation debe ser 'mean' o 'max'.")

        report[group] = {
            "support": float(group_support(sensitive, group)),
            "disparity": disparity_value,
        }

    return report


def group_class_disparity_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
) -> dict[Any, dict[Any, dict[str, float]]]:
    """
    Genera un informe detallado por grupo y por clase en esquema one-vs-rest.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.

    Retorna
    -------
    dict[Any, dict[Any, dict[str, float]]]
        Diccionario anidado con métricas y disparidad por grupo y clase.
    """
    y_true, y_pred, sensitive = validate_fairness_inputs(y_true, y_pred, sensitive)
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    report: dict[Any, dict[Any, dict[str, float]]] = {}
    classes = unique_classes(y_true, y_pred)

    for group in unique_sensitive_groups(sensitive):
        report[group] = {}

        for target_class in classes:
            metrics = group_class_fairness_metrics(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
                group=group,
                target_class=target_class,
            )

            spd_v = spd_violation(metrics["spd"], thresholds.spd)
            eod_v = eod_violation(metrics["eod"], thresholds.eod)
            di_v = di_violation(metrics["di_sym"], thresholds.di)

            disparity = (
                weights.spd * spd_v +
                weights.eod * eod_v +
                weights.di * di_v
            )

            if weights.normalize:
                _total = active_weight_sum(weights)
                disparity = disparity / _total if _total > 0 else 0.0

            report[group][target_class] = {
                "spd": float(metrics["spd"]),
                "eod": float(metrics["eod"]),
                "di_sym": float(metrics["di_sym"]),
                "spd_violation": float(spd_v),
                "eod_violation": float(eod_v),
                "di_violation": float(di_v),
                "disparity": float(disparity),
            }

    return report