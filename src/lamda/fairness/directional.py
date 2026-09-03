"""
Disparidad direccional por grupo y clase.

Motivación
----------
La disparidad agregada empleada por Fair-GAD, D_g, se construye a partir de las
violaciones de umbral de SPD, EOD e impacto dispar simétrico, todas ellas
definidas sobre valores absolutos o simetrizados. Cuando el atributo sensible
define exactamente dos grupos, esa construcción hace que la disparidad sea
idéntica en ambos, por una razón estructural:

    SPD(A, c) = P(c | A) - P(c | no A) = -[P(c | B) - P(c | no B)] = -SPD(B, c)
    EOD(A, c) = -EOD(B, c)
    DI_sim(A, c) = min(DI, 1/DI) = DI_sim(B, c)

Al tomar valor absoluto de las dos primeras y simetrizar la tercera, los dos
grupos reciben el mismo número. En consecuencia, la penalización
lambda * D_{g(r)} es la misma constante para todo individuo y el mecanismo no
distingue entre el grupo favorecido y el desfavorecido: corrige de forma
indirecta, invirtiendo las decisiones cuyo margen es menor que esa constante.

Este módulo define una disparidad **direccional**, que conserva el signo y se
indexa por par de grupo y clase:

    D_dir(g, c) = [ w_spd * max(0, SPD(g, c) - tau_spd)
                  + w_eod * max(0, EOD(g, c) - tau_eod) ] / (w_spd + w_eod)

Solo es positiva cuando el grupo g está **sobreasignado** a la clase c por
encima del umbral. El grupo infrarrepresentado en esa clase recibe cero, de modo
que la penalización pasa a ser específica del par y direccionalmente correcta.

El impacto dispar queda fuera de esta composición. Su versión simétrica es, por
construcción, invariante al grupo, y su versión habitual es un cociente cuya
escala no es comparable con la de las dos diferencias. Como efecto colateral,
la formulación direccional resulta idéntica en tareas binarias y multiclase, lo
que elimina la asimetría entre perfiles de métricas.

Este módulo no modifica el comportamiento del Fair-GAD original: define las
funciones que la variante direccional necesita, de modo que ambas versiones
puedan coexistir y compararse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lamda.fairness.disparity import (
    FairnessThresholds,
    FairnessWeights,
    class_support,
    group_support,
    unique_classes,
    unique_sensitive_groups,
    validate_fairness_inputs,
    validate_thresholds,
    validate_weights,
)
from lamda.fairness.metrics import (
    equal_opportunity_difference,
    statistical_parity_difference,
)


# =========================================================
# VIOLACIONES DIRECCIONALES
# =========================================================

def directional_violation(
    value: float,
    tau: float,
) -> float:
    """
    Calcula la violación direccional de un umbral.

    A diferencia de la violación simétrica, que emplea el valor absoluto, esta
    función solo devuelve un valor positivo cuando la métrica supera el umbral
    en sentido positivo, es decir, cuando el grupo está sobreasignado a la
    clase evaluada. En caso contrario devuelve cero.

    Parámetros
    ----------
    value : float
        Valor de la métrica, con su signo.
    tau : float
        Umbral de referencia.

    Retorna
    -------
    float
        Exceso no negativo del umbral en sentido positivo.
    """
    return max(0.0, float(value) - float(tau))


def directional_weight_sum(weights: FairnessWeights) -> float:
    """
    Suma de los pesos activos en la composición direccional.

    El impacto dispar no interviene en esta composición, de modo que la suma
    considera únicamente los pesos de las dos diferencias.

    Parámetros
    ----------
    weights : FairnessWeights
        Pesos de agregación.

    Retorna
    -------
    float
        Suma de los pesos activos.
    """
    return float(weights.spd) + float(weights.eod)


# =========================================================
# DISPARIDAD DIRECCIONAL DE UN PAR GRUPO-CLASE
# =========================================================

def directional_group_class_disparity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    target_class: Any,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
) -> float:
    """
    Calcula la disparidad direccional D_dir(g, c) de un par de grupo y clase.

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
        Clase evaluada bajo el esquema de una clase frente al resto.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación. El peso del impacto dispar se ignora.

    Retorna
    -------
    float
        Disparidad direccional, no negativa. Es cero cuando el grupo no está
        sobreasignado a la clase por encima del umbral.
    """
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

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

    valor = (
        weights.spd * directional_violation(spd, thresholds.spd)
        + weights.eod * directional_violation(eod, thresholds.eod)
    )

    # La normalización mantiene la escala comparable entre configuraciones,
    # igual que en la composición simétrica.
    if weights.normalize:
        total = directional_weight_sum(weights)
        valor = valor / total if total > 0 else 0.0

    return float(valor)


# =========================================================
# DISPARIDAD DIRECCIONAL DE TODOS LOS PARES
# =========================================================

def all_directional_disparities(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    min_group_size: int = 1,
    min_positive_size: int = 1,
    default_for_small_groups: float = 0.0,
) -> dict[tuple[Any, Any], float]:
    """
    Calcula la disparidad direccional de todos los pares de grupo y clase.

    Los filtros de soporte replican los de la disparidad simétrica: un grupo
    con menos individuos que min_group_size recibe el valor por defecto en
    todas sus clases, y un par de grupo y clase con menos ejemplos reales de esa
    clase que min_positive_size recibe también el valor por defecto, ya que su
    diferencia de igualdad de oportunidad no sería estimable de forma estable.

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
        Soporte mínimo del grupo.
    min_positive_size : int, default=1
        Soporte mínimo de ejemplos reales de la clase dentro del grupo.
    default_for_small_groups : float, default=0.0
        Valor asignado a los pares sin soporte suficiente.

    Retorna
    -------
    dict[tuple[Any, Any], float]
        Diccionario {(grupo, clase): D_dir}.
    """
    y_true, y_pred, sensitive = validate_fairness_inputs(y_true, y_pred, sensitive)
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    disparidades: dict[tuple[Any, Any], float] = {}
    clases = unique_classes(y_true, y_pred)

    for group in unique_sensitive_groups(sensitive):
        n_group = group_support(sensitive, group)

        for target_class in clases:
            if n_group < min_group_size:
                disparidades[(group, target_class)] = float(default_for_small_groups)
                continue

            n_positive = class_support(
                y_true=y_true,
                sensitive=sensitive,
                group=group,
                target_class=target_class,
            )
            if n_positive < min_positive_size:
                disparidades[(group, target_class)] = float(default_for_small_groups)
                continue

            disparidades[(group, target_class)] = directional_group_class_disparity(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
                group=group,
                target_class=target_class,
                thresholds=thresholds,
                weights=weights,
            )

    return disparidades


# =========================================================
# MATRIZ DE PENALIZACIÓN Y REGULARIZACIÓN DEL GAD
# =========================================================

def directional_penalty_matrix(
    sensitive: np.ndarray,
    classes: np.ndarray,
    disparities: dict[tuple[Any, Any], float],
    default_group_disparity: float = 0.0,
) -> np.ndarray:
    """
    Construye la matriz de penalización de dimensiones (n_individuos, n_clases).

    Cada posición contiene la disparidad direccional del par formado por el
    grupo del individuo y la clase de la columna. Los pares no observados
    durante el entrenamiento reciben el valor por defecto.

    Parámetros
    ----------
    sensitive : np.ndarray
        Vector de grupos sensibles de los individuos a predecir.
    classes : np.ndarray
        Clases del modelo, en el mismo orden que las columnas del GAD.
    disparities : dict[tuple[Any, Any], float]
        Disparidades direccionales almacenadas.
    default_group_disparity : float, default=0.0
        Valor para pares no observados.

    Retorna
    -------
    np.ndarray
        Matriz de penalización.
    """
    sensitive = np.asarray(sensitive).reshape(-1)
    classes = np.asarray(classes).reshape(-1)

    matriz = np.full((sensitive.shape[0], classes.shape[0]),
                     float(default_group_disparity), dtype=float)

    for j, c in enumerate(classes):
        for group in np.unique(sensitive):
            valor = disparities.get((group, c), float(default_group_disparity))
            matriz[sensitive == group, j] = float(valor)

    return matriz


def regularize_gad_directional(
    gad: np.ndarray,
    sensitive: np.ndarray,
    classes: np.ndarray,
    disparities: dict[tuple[Any, Any], float],
    lambda_: float,
    clip: bool = False,
    default_group_disparity: float = 0.0,
    only_preliminary_class: bool = True,
) -> np.ndarray:
    """
    Aplica la regularización direccional sobre la matriz de GAD.

    Con only_preliminary_class activado, que es el comportamiento por defecto y
    el que preserva la estructura de la formulación original, la penalización
    afecta únicamente a la clase preliminarmente seleccionada:

        GAD_fair(c, r) = GAD(c, r) - lambda * D_dir(g(r), c) * 1[c = c_pre(r)]

    Con la opción desactivada, la penalización se aplica a todas las clases, lo
    que constituye una reordenación completa de la puntuación por clase:

        GAD_fair(c, r) = GAD(c, r) - lambda * D_dir(g(r), c)

    Parámetros
    ----------
    gad : np.ndarray
        Matriz de grados de adecuación global, de dimensiones
        (n_individuos, n_clases).
    sensitive : np.ndarray
        Vector de grupos sensibles.
    classes : np.ndarray
        Clases del modelo, en el orden de las columnas de gad.
    disparities : dict[tuple[Any, Any], float]
        Disparidades direccionales almacenadas.
    lambda_ : float
        Intensidad de la penalización.
    clip : bool, default=False
        Si es True, recorta el resultado al intervalo [0, 1].
    default_group_disparity : float, default=0.0
        Valor para pares no observados.
    only_preliminary_class : bool, default=True
        Restringe la penalización a la clase preliminar.

    Retorna
    -------
    np.ndarray
        Matriz de GAD regularizado.
    """
    gad = np.asarray(gad, dtype=float)
    penalizacion = directional_penalty_matrix(
        sensitive=sensitive,
        classes=classes,
        disparities=disparities,
        default_group_disparity=default_group_disparity,
    )

    if only_preliminary_class:
        # La clase preliminar es la que maximiza el GAD sin penalizar.
        indicador = np.zeros_like(gad, dtype=float)
        preliminar = np.argmax(gad, axis=1)
        indicador[np.arange(gad.shape[0]), preliminar] = 1.0
        penalizacion = penalizacion * indicador

    gad_fair = gad - float(lambda_) * penalizacion

    if clip:
        gad_fair = np.clip(gad_fair, 0.0, 1.0)

    return gad_fair


# =========================================================
# INFORME
# =========================================================

def directional_disparity_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
) -> list[dict[str, Any]]:
    """
    Genera el detalle por grupo y clase de la disparidad direccional.

    Permite comprobar de forma directa que los valores difieren entre grupos,
    a diferencia de lo que ocurre con la composición simétrica.

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
    list[dict[str, Any]]
        Una entrada por par de grupo y clase, con las métricas con signo y la
        disparidad direccional resultante.
    """
    y_true, y_pred, sensitive = validate_fairness_inputs(y_true, y_pred, sensitive)
    thresholds = validate_thresholds(thresholds)
    weights = validate_weights(weights)

    filas: list[dict[str, Any]] = []

    for group in unique_sensitive_groups(sensitive):
        for target_class in unique_classes(y_true, y_pred):
            spd = statistical_parity_difference(
                y_pred=y_pred, sensitive=sensitive,
                group=group, positive_label=target_class,
            )
            eod = equal_opportunity_difference(
                y_true=y_true, y_pred=y_pred, sensitive=sensitive,
                group=group, positive_label=target_class,
            )
            filas.append({
                "group": group,
                "target_class": target_class,
                "spd": float(spd),
                "eod": float(eod),
                "exceso_spd": directional_violation(spd, thresholds.spd),
                "exceso_eod": directional_violation(eod, thresholds.eod),
                "disparidad_direccional": directional_group_class_disparity(
                    y_true=y_true, y_pred=y_pred, sensitive=sensitive,
                    group=group, target_class=target_class,
                    thresholds=thresholds, weights=weights,
                ),
            })

    return filas


# =========================================================
# AGREGACIÓN POR GRUPO
# =========================================================

def all_directional_group_disparities(
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
    Agrega la disparidad direccional por grupo, sobre el conjunto de clases.

    Es el análogo direccional de la disparidad agregada por grupo y se emplea
    en la variante direccional de Fair-MAD, donde el impacto de un descriptor
    se define como la diferencia entre la disparidad del grupo con el modelo
    completo y sin ese descriptor. A diferencia de la versión simétrica, esta
    agregación sí toma valores distintos en cada grupo, de modo que el paso de
    peor caso entre grupos deja de ser vacío.

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
        Soporte mínimo del grupo.
    min_positive_size : int, default=1
        Soporte mínimo de ejemplos reales de la clase dentro del grupo.
    default_for_small_groups : float, default=0.0
        Valor asignado a los grupos sin soporte suficiente.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases, "mean" o "max".

    Retorna
    -------
    dict[Any, float]
        Diccionario {grupo: D_dir_g}.
    """
    if class_aggregation not in ("mean", "max"):
        raise ValueError("class_aggregation debe ser 'mean' o 'max'.")

    pares = all_directional_disparities(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        thresholds=thresholds,
        weights=weights,
        min_group_size=min_group_size,
        min_positive_size=min_positive_size,
        default_for_small_groups=default_for_small_groups,
    )

    por_grupo: dict[Any, list[float]] = {}
    for (group, _clase), valor in pares.items():
        por_grupo.setdefault(group, []).append(float(valor))

    agregadas: dict[Any, float] = {}
    for group, valores in por_grupo.items():
        if not valores:
            agregadas[group] = float(default_for_small_groups)
        elif class_aggregation == "mean":
            agregadas[group] = float(np.mean(valores))
        else:
            agregadas[group] = float(np.max(valores))

    return agregadas
