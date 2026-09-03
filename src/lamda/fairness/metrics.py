"""
Métricas de equidad por grupo sensible.

Este módulo implementa métricas de fairness utilizadas para evaluar la
disparidad entre un grupo sensible concreto y el resto de la población.

Métricas implementadas
----------------------
1. Statistical Parity Difference (SPD)
   SPD(g) = P(Y_hat = 1 | S = g) - P(Y_hat = 1 | S != g)

2. Equal Opportunity Difference (EOD)
   EOD(g) = P(Y_hat = 1 | Y = 1, S = g) - P(Y_hat = 1 | Y = 1, S != g)

3. Disparate Impact (DI)
   DI(g) = P(Y_hat = 1 | S = g) / P(Y_hat = 1 | S != g)

4. Disparate Impact Simétrico (DI_sym)
   DI_sym(g) = min(DI(g), 1 / DI(g))

Métricas multiclase (reporte comparativo)
-----------------------------------------
5. SPD multiclase (one-vs-rest agregado)
6. MEO (Mean/Max Equalized Odds) al estilo Alghamdi et al. (2022)

Las métricas multiclase se emplean únicamente en las tablas de detección para
hacerlas comparables con la literatura de equidad multiclase. No intervienen en
el cálculo de la disparidad D_g ni en la penalización de Fair-GAD o Fair-MAD;
el escaparate comparable y el motor de penalización se mantienen separados.

Interpretación
--------------
- SPD y EOD tienen valor ideal 0.
- DI y DI_sym tienen valor ideal 1.
- DI_sym permite expresar de forma simétrica la desviación respecto a la equidad.
- SPD multiclase y MEO tienen valor ideal 0; valores mayores indican más disparidad.

Flujo
-----
1. Se define un grupo sensible g.
2. Se compara dicho grupo con el complemento de la población.
3. Se calcula la tasa de resultado favorable en cada subconjunto.
4. Se construyen las métricas de equidad a partir de esas tasas.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =========================================================
# VALIDACIÓN
# =========================================================

def validate_metric_inputs(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    y_true: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Valida las entradas utilizadas en el cálculo de métricas de equidad.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    y_true : np.ndarray | None, default=None
        Etiquetas reales. Solo es necesario para métricas que las requieran.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        Entradas convertidas a arrays unidimensionales y validadas.

    Lanza
    ------
    ValueError
        Si las longitudes de los arrays no coinciden.
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)

    if y_true is not None:
        y_true = np.asarray(y_true).reshape(-1)
        if not (len(y_true) == len(y_pred) == len(sensitive)):
            raise ValueError("y_true, y_pred y sensitive deben tener la misma longitud.")
    else:
        if len(y_pred) != len(sensitive):
            raise ValueError("y_pred y sensitive deben tener la misma longitud.")

    return y_pred, sensitive, y_true


# =========================================================
# UTILIDADES
# =========================================================

def group_mask(
    sensitive: np.ndarray,
    group: Any,
) -> np.ndarray:
    """
    Construye la máscara booleana correspondiente a un grupo sensible.

    Parámetros
    ----------
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a seleccionar.

    Retorna
    -------
    np.ndarray
        Máscara booleana del grupo.
    """
    sensitive = np.asarray(sensitive).reshape(-1)
    return sensitive == group


def complement_mask(
    sensitive: np.ndarray,
    group: Any,
) -> np.ndarray:
    """
    Construye la máscara booleana correspondiente al complemento de un grupo.

    Parámetros
    ----------
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo cuya población complementaria se desea obtener.

    Retorna
    -------
    np.ndarray
        Máscara booleana del complemento del grupo.
    """
    sensitive = np.asarray(sensitive).reshape(-1)
    return sensitive != group


def safe_mean(
    values: np.ndarray,
    default: float = 0.0,
) -> float:
    """
    Calcula la media de un vector manejando el caso vacío.

    Parámetros
    ----------
    values : np.ndarray
        Valores sobre los que se calcula la media.
    default : float, default=0.0
        Valor a devolver si el vector está vacío.

    Retorna
    -------
    float
        Media del vector o valor por defecto.
    """
    values = np.asarray(values)

    if values.size == 0:
        return float(default)

    return float(np.mean(values))


def favorable_rate(
    y_pred: np.ndarray,
    mask: np.ndarray,
    positive_label: int | str = 1,
    default: float = 0.0,
) -> float:
    """
    Calcula la tasa de predicciones favorables dentro de una subpoblación.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    mask : np.ndarray
        Máscara de la subpoblación.
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default : float, default=0.0
        Valor devuelto si la subpoblación está vacía.

    Retorna
    -------
    float
        Tasa de predicción favorable en la subpoblación.
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.asarray(mask).reshape(-1)

    if np.sum(mask) == 0:
        return float(default)

    return safe_mean(y_pred[mask] == positive_label, default=default)


def true_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    positive_label: int | str = 1,
    default: float = 0.0,
) -> float:
    """
    Calcula la tasa de verdaderos positivos dentro de una subpoblación.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    mask : np.ndarray
        Máscara de la subpoblación.
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default : float, default=0.0
        Valor devuelto si no hay positivos reales en la subpoblación.

    Retorna
    -------
    float
        Tasa de verdaderos positivos en la subpoblación.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.asarray(mask).reshape(-1)

    positive_mask = mask & (y_true == positive_label)

    if np.sum(positive_mask) == 0:
        return float(default)

    return safe_mean(y_pred[positive_mask] == positive_label, default=default)


def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    positive_label: int | str = 1,
    default: float = 0.0,
) -> float:
    """
    Calcula la tasa de falsos positivos dentro de una subpoblación.

    En esquema one-vs-rest respecto a positive_label, los negativos reales son
    los individuos de la máscara cuya etiqueta real no es la clase favorable, y
    la FPR mide qué proporción de ellos se predice como la clase favorable.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    mask : np.ndarray
        Máscara de la subpoblación.
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default : float, default=0.0
        Valor devuelto si no hay negativos reales en la subpoblación.

    Retorna
    -------
    float
        Tasa de falsos positivos en la subpoblación.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.asarray(mask).reshape(-1)

    negative_mask = mask & (y_true != positive_label)

    if np.sum(negative_mask) == 0:
        return float(default)

    return safe_mean(y_pred[negative_mask] == positive_label, default=default)


# =========================================================
# MÉTRICAS PRINCIPALES
# =========================================================

def statistical_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    positive_label: int | str = 1,
    default_group_rate: float = 0.0,
    default_reference_rate: float = 0.0,
) -> float:
    """
    Calcula Statistical Parity Difference (SPD) para un grupo sensible.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default_group_rate : float, default=0.0
        Valor por defecto si el grupo está vacío.
    default_reference_rate : float, default=0.0
        Valor por defecto si el complemento del grupo está vacío.

    Retorna
    -------
    float
        Valor de SPD del grupo.
    """
    y_pred, sensitive, _ = validate_metric_inputs(y_pred=y_pred, sensitive=sensitive)

    mask_group = group_mask(sensitive, group)
    mask_reference = complement_mask(sensitive, group)

    p_group = favorable_rate(
        y_pred=y_pred,
        mask=mask_group,
        positive_label=positive_label,
        default=default_group_rate,
    )
    p_reference = favorable_rate(
        y_pred=y_pred,
        mask=mask_reference,
        positive_label=positive_label,
        default=default_reference_rate,
    )

    return float(p_group - p_reference)


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    positive_label: int | str = 1,
    default_group_tpr: float = 0.0,
    default_reference_tpr: float = 0.0,
) -> float:
    """
    Calcula Equal Opportunity Difference (EOD) para un grupo sensible.

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
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default_group_tpr : float, default=0.0
        Valor por defecto si el grupo no tiene positivos reales.
    default_reference_tpr : float, default=0.0
        Valor por defecto si el complemento no tiene positivos reales.

    Retorna
    -------
    float
        Valor de EOD del grupo.
    """
    y_pred, sensitive, y_true = validate_metric_inputs(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
    )

    mask_group = group_mask(sensitive, group)
    mask_reference = complement_mask(sensitive, group)

    tpr_group = true_positive_rate(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask_group,
        positive_label=positive_label,
        default=default_group_tpr,
    )
    tpr_reference = true_positive_rate(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask_reference,
        positive_label=positive_label,
        default=default_reference_tpr,
    )

    return float(tpr_group - tpr_reference)


def disparate_impact(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    positive_label: int | str = 1,
    default_value: float = 1.0,
) -> float:
    """
    Calcula Disparate Impact (DI) para un grupo sensible.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default_value : float, default=1.0
        Valor devuelto cuando el cálculo no es evaluable porque la clase
        favorable no aparece en ningún subgrupo (denominador y numerador
        ambos cero). Se usa 1.0 porque la ausencia de la clase en todos
        los grupos no indica disparidad entre ellos.

    Retorna
    -------
    float
        Valor de DI del grupo.
    """
    y_pred, sensitive, _ = validate_metric_inputs(y_pred=y_pred, sensitive=sensitive)

    mask_group = group_mask(sensitive, group)
    mask_reference = complement_mask(sensitive, group)

    p_group = favorable_rate(
        y_pred=y_pred,
        mask=mask_group,
        positive_label=positive_label,
        default=0.0,
    )
    p_reference = favorable_rate(
        y_pred=y_pred,
        mask=mask_reference,
        positive_label=positive_label,
        default=0.0,
    )

    if p_reference == 0.0:
        # Si el complemento no predice la clase favorable, el cociente es
        # indefinido. Distinguimos dos situaciones:
        # - p_group == 0 también: la clase no aparece en ningún subgrupo.
        #   No hay información para medir disparidad → valor neutro (1.0).
        # - p_group > 0: el grupo sí recibe la clase pero el complemento no.
        #   Es el caso inverso de discriminación, pero DI no está definido
        #   algebraicamente → se devuelve también 1.0 para evitar artefactos,
        #   aunque SPD y EOD captarán esta asimetría igualmente.
        return float(default_value)

    return float(p_group / p_reference)


def disparate_impact_symmetric(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    positive_label: int | str = 1,
    default_value: float = 1.0,
) -> float:
    """
    Calcula Disparate Impact simétrico para un grupo sensible.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar.
    positive_label : int | str, default=1
        Etiqueta considerada como favorable.
    default_value : float, default=1.0
        Valor devuelto cuando DI no es evaluable. Se usa 1.0 (equidad
        perfecta) para evitar inflar D_g en clases raras del esquema
        one-vs-rest donde la clase favorable no aparece en ningún subgrupo.

    Retorna
    -------
    float
        Valor de DI simétrico del grupo. Vale 1.0 en caso de equidad
        perfecta o cuando el cálculo no es evaluable. Valores menores
        indican mayor disparidad.
    """
    di = disparate_impact(
        y_pred=y_pred,
        sensitive=sensitive,
        group=group,
        positive_label=positive_label,
        default_value=default_value,
    )

    if di <= 0.0:
        # DI = 0 ocurre cuando p_group = 0 y p_reference > 0: el grupo no
        # recibe nunca la clase favorable pero el complemento sí. Es el caso
        # de mayor desventaja posible; di_sym = 0 es correcto.
        return 0.0

    return float(min(di, 1.0 / di))


# =========================================================
# MÉTRICA INTERSECCIONAL AGREGADA (HFI)
# =========================================================

def harmonic_fairness_intersectional(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    positive_label: int | str = 1,
    groups: list | None = None,
    drop_zero_groups: bool = False,
) -> float:
    """
    Calcula la métrica HFI (Harmonic Fairness Intersectional).

    HFI es la media armónica, sobre los N grupos interseccionales, de la
    equidad proporcional de cada grupo medida como min(DI_g, 1/DI_g), es
    decir, el impacto dispar simétrico ya implementado en este módulo:

        HFI = N / sum_g ( 1 / di_sym(g) )

    Su valor ideal es 1 y el umbral de referencia es 0,8. La media armónica
    hace que un solo subgrupo con equidad proporcional baja arrastre el valor
    agregado, que es la propiedad por la que la métrica se propone para el
    análisis interseccional.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles, posiblemente compuesto.
    positive_label : int | str, default=1
        Etiqueta considerada favorable. En esquema uno contra el resto se
        invoca una vez por clase.
    groups : list | None, default=None
        Grupos a considerar. Si es None se toman todos los presentes.
    drop_zero_groups : bool, default=False
        Si False se aplica la definición literal: un grupo con di_sym = 0
        (nunca recibe la clase favorable mientras el complemento sí) lleva el
        agregado a 0. Si True esos grupos se excluyen del promedio y N se
        ajusta, lo que produce un valor definido cuando algún subgrupo queda
        completamente sin asignación favorable.

    Retorna
    -------
    float
        Valor de HFI en [0, 1].
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    if groups is None:
        groups = sorted(set(sensitive.tolist()))

    scores = [
        disparate_impact_symmetric(
            y_pred=y_pred,
            sensitive=sensitive,
            group=g,
            positive_label=positive_label,
        )
        for g in groups
    ]

    if drop_zero_groups:
        scores = [s for s in scores if s > 0.0]
        if not scores:
            return 0.0
    elif any(s <= 0.0 for s in scores):
        # Definición literal: la media armónica se anula si un solo subgrupo
        # no recibe nunca el resultado favorable.
        return 0.0

    return float(len(scores) / sum(1.0 / s for s in scores))


def hfi_one_vs_rest(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    classes: np.ndarray | None = None,
    groups: list | None = None,
    drop_zero_groups: bool = False,
) -> dict:
    """
    Calcula HFI por clase en esquema uno contra el resto.

    HFI se define sobre un único resultado favorable. En problemas multiclase
    se aplica la misma descomposición uno contra el resto que emplea el resto
    del protocolo: cada clase se toma sucesivamente como resultado favorable y
    se obtiene un HFI por clase. La agregación sobre clases se deja al
    llamante, que puede reportar el mínimo, el más restrictivo, y la media.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    classes : np.ndarray | None, default=None
        Clases a recorrer. Si es None se toman las presentes en y_pred.
    groups : list | None, default=None
        Grupos a considerar. Si es None se toman todos los presentes.
    drop_zero_groups : bool, default=False
        Se traslada a harmonic_fairness_intersectional.

    Retorna
    -------
    dict
        Diccionario {clase: HFI de esa clase}.
    """
    y_pred = np.asarray(y_pred)

    if classes is None:
        classes = np.unique(y_pred)

    return {
        c: harmonic_fairness_intersectional(
            y_pred=y_pred,
            sensitive=sensitive,
            positive_label=c,
            groups=groups,
            drop_zero_groups=drop_zero_groups,
        )
        for c in classes
    }


# =========================================================
# MÉTRICAS MULTICLASE (REPORTE COMPARATIVO)
# =========================================================

def spd_multiclass(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    classes: np.ndarray | None = None,
    aggregation: str = "mean",
) -> float:
    """
    Calcula el SPD multiclase para un grupo sensible en esquema one-vs-rest.

    Para cada clase c se calcula |SPD_c| tratando c como resultado favorable,
    y se agrega sobre clases. Se emplea únicamente para el reporte comparativo
    de detección; no interviene en la disparidad D_g.

    Parámetros
    ----------
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar frente al complemento.
    classes : np.ndarray | None, default=None
        Conjunto de clases a considerar. Si es None, se infiere de y_pred.
    aggregation : str, default="mean"
        Estrategia de agregación sobre clases. Opciones:
        - "mean": promedio de |SPD_c| (SPD multiclase habitual)
        - "max": máximo de |SPD_c|

    Retorna
    -------
    float
        SPD multiclase agregado para el grupo.
    """
    y_pred, sensitive, _ = validate_metric_inputs(y_pred=y_pred, sensitive=sensitive)

    if classes is None:
        classes = np.unique(y_pred)

    per_class = []

    for c in classes:
        spd_c = statistical_parity_difference(
            y_pred=y_pred,
            sensitive=sensitive,
            group=group,
            positive_label=c,
        )
        per_class.append(abs(spd_c))

    if len(per_class) == 0:
        return 0.0

    if aggregation == "mean":
        return float(np.mean(per_class))

    if aggregation == "max":
        return float(np.max(per_class))

    raise ValueError("aggregation debe ser 'mean' o 'max'.")


def mean_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group: Any,
    classes: np.ndarray | None = None,
) -> float:
    """
    Calcula el MEO (equalized odds multiclase por máximo) para un grupo sensible
    frente a su complemento, siguiendo la convención de Alghamdi et al. (2022).

    Para cada clase c se calcula la semisuma de la diferencia de TPR y de FPR
    entre el grupo y el complemento, y se devuelve el máximo sobre clases:

        term_c = ( |TPR_c(grupo) - TPR_c(resto)|
                 + |FPR_c(grupo) - FPR_c(resto)| ) / 2
        MEO = max_c term_c

    Nota de comparabilidad: Alghamdi et al. definen MEO como máximo sobre clases
    y sobre pares de grupos. Aquí el atributo sensible se evalúa en esquema
    grupo-vs-resto, coherente con el resto del paquete. Con atributo sensible
    binario ambas definiciones coinciden; con más de dos grupos se reporta el
    máximo sobre (clase, grupo-vs-resto).

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group : Any
        Grupo a evaluar frente al complemento.
    classes : np.ndarray | None, default=None
        Conjunto de clases a considerar. Si es None, se infiere de y_true e y_pred.

    Retorna
    -------
    float
        Valor de MEO para el grupo (máximo sobre clases).
    """
    y_pred, sensitive, y_true = validate_metric_inputs(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
    )

    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))

    mask_group = group_mask(sensitive, group)
    mask_reference = complement_mask(sensitive, group)

    per_class_terms = []

    for c in classes:
        tpr_group = true_positive_rate(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask_group,
            positive_label=c,
        )
        tpr_reference = true_positive_rate(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask_reference,
            positive_label=c,
        )
        fpr_group = false_positive_rate(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask_group,
            positive_label=c,
        )
        fpr_reference = false_positive_rate(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask_reference,
            positive_label=c,
        )

        term_c = (abs(tpr_group - tpr_reference) + abs(fpr_group - fpr_reference)) / 2.0
        per_class_terms.append(term_c)

    if len(per_class_terms) == 0:
        return 0.0

    return float(np.max(per_class_terms))


def multiclass_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    classes: np.ndarray | None = None,
    spd_aggregation: str = "mean",
) -> dict[Any, dict[str, float]]:
    """
    Genera el reporte crudo multiclase por grupo sensible, para las tablas de
    detección comparables con la literatura (SPD multiclase y MEO).

    Estas métricas no intervienen en el cálculo de D_g; constituyen el
    escaparate comparable de la fase de detección.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    classes : np.ndarray | None, default=None
        Conjunto de clases. Si es None, se infiere de y_true e y_pred.
    spd_aggregation : str, default="mean"
        Estrategia de agregación del SPD multiclase ("mean" o "max").

    Retorna
    -------
    dict[Any, dict[str, float]]
        Diccionario {grupo: {"spd_multiclass": ..., "meo": ...}}.
    """
    y_pred, sensitive, y_true = validate_metric_inputs(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
    )

    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))

    report: dict[Any, dict[str, float]] = {}

    for group in np.unique(sensitive):
        report[group] = {
            "spd_multiclass": spd_multiclass(
                y_pred=y_pred,
                sensitive=sensitive,
                group=group,
                classes=classes,
                aggregation=spd_aggregation,
            ),
            "meo": mean_equalized_odds(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
                group=group,
                classes=classes,
            ),
        }

    return report
