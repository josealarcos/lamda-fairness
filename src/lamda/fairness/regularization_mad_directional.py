"""
Regularización direccional sobre MAD.

Variante de la regularización por descriptor que emplea la disparidad
direccional como señal de control, en lugar de la disparidad simétrica.

Motivación
----------
La formulación de Fair-MAD define el impacto de un descriptor como la
diferencia entre la disparidad del grupo con el modelo completo y la que
resulta al retirar ese descriptor de la agregación, y toma después el peor caso
entre grupos:

    C_{j,g} = D_g - D_{g,(-j)}
    C_j     = max_g C_{j,g}
    D_j     = max(0, C_j)
    w_j     = 1 / (1 + eta * max(0, D_j - tau_j))

Cuando el atributo sensible define exactamente dos grupos, la disparidad
simétrica D_g toma el mismo valor en ambos, por antisimetría de las métricas que
la componen. En consecuencia, C_{j,g} tampoco depende del grupo y el paso de
peor caso entre grupos es formalmente vacío: selecciona un máximo sobre valores
idénticos.

Este módulo sustituye D_g por su versión direccional, que sí toma valores
distintos en cada grupo porque solo acumula el exceso de umbral en el sentido de
la sobreasignación. Con ella, C_{j,g} difiere entre grupos y el peor caso
selecciona efectivamente el grupo al que ese descriptor perjudica más, que es lo
que la formulación pretendía expresar.

El resto de la propuesta permanece inalterado: el peso por descriptor sigue
siendo único y global, aplicado a todos los individuos con independencia de su
grupo, y la regularización sigue actuando como exponente sobre el grado de
adecuación marginal. La disparidad simétrica se conserva como magnitud de
reporte.

Este módulo no modifica el comportamiento de la regularización original.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lamda.core.decision import predict_class_labels
from lamda.fairness.directional import all_directional_group_disparities
from lamda.fairness.disparity import FairnessThresholds, FairnessWeights
from lamda.fairness.regularization_mad import (
    aggregate_mad_to_gad,
    apply_mad_fairness,
    compute_descriptor_disparity_from_group_impact,
    compute_descriptor_group_impact,
    compute_descriptor_weights,
    predictions_from_mad,
    remove_descriptor_from_mad,
    validate_alpha,
    validate_mad_tensor,
    validate_operator,
)


# =========================================================
# IMPACTO DIRECCIONAL POR DESCRIPTOR
# =========================================================

def compute_all_descriptor_disparities_directional(
    mad: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    classes: np.ndarray,
    alpha: float = 0.5,
    operator: str = "product",
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    min_group_size: int = 1,
    min_positive_size: int = 1,
    default_for_small_groups: float = 0.0,
    class_aggregation: str = "mean",
) -> tuple[np.ndarray, list[dict[Any, float]], dict[Any, float]]:
    """
    Calcula la disparidad efectiva de todos los descriptores con señal direccional.

    Reproduce el procedimiento de la versión simétrica, retirando cada
    descriptor de la agregación y midiendo la variación de la disparidad, con la
    única diferencia de que la disparidad por grupo se obtiene mediante la
    agregación direccional.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor MAD del modelo completo.
    y_true : np.ndarray
        Etiquetas reales.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    classes : np.ndarray
        Vector de clases del modelo.
    alpha : float, default=0.5
        Parámetro de exigencia de LAMDA.
    operator : str, default="product"
        Operador de agregación empleado en el grado de adecuación global.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    min_group_size : int, default=1
        Soporte mínimo del grupo.
    min_positive_size : int, default=1
        Soporte mínimo por clase para la diferencia de igualdad de oportunidad.
    default_for_small_groups : float, default=0.0
        Valor asignado a grupos con soporte insuficiente.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.

    Retorna
    -------
    tuple[np.ndarray, list[dict[Any, float]], dict[Any, float]]
        Vector de disparidades por descriptor, lista de impactos por grupo de
        cada descriptor y disparidad direccional por grupo del modelo completo.
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)
    validate_operator(operator)

    n_descriptors = mad.shape[2]

    baseline_pred = predictions_from_mad(
        mad=mad,
        classes=classes,
        alpha=alpha,
        operator=operator,
    )

    baseline_group_disparities = all_directional_group_disparities(
        y_true=y_true,
        y_pred=baseline_pred,
        sensitive=sensitive,
        thresholds=thresholds,
        weights=weights,
        min_group_size=min_group_size,
        min_positive_size=min_positive_size,
        default_for_small_groups=default_for_small_groups,
        class_aggregation=class_aggregation,
    )

    descriptor_disparities = np.zeros(n_descriptors, dtype=float)
    descriptor_group_impacts: list[dict[Any, float]] = []

    for j in range(n_descriptors):
        mad_reduced = remove_descriptor_from_mad(mad=mad, descriptor_index=j)

        reduced_pred = predictions_from_mad(
            mad=mad_reduced,
            classes=classes,
            alpha=alpha,
            operator=operator,
        )

        reduced_group_disparities = all_directional_group_disparities(
            y_true=y_true,
            y_pred=reduced_pred,
            sensitive=sensitive,
            thresholds=thresholds,
            weights=weights,
            min_group_size=min_group_size,
            min_positive_size=min_positive_size,
            default_for_small_groups=default_for_small_groups,
            class_aggregation=class_aggregation,
        )

        group_impact = compute_descriptor_group_impact(
            baseline_group_disparities=baseline_group_disparities,
            reduced_group_disparities=reduced_group_disparities,
        )

        descriptor_group_impacts.append(group_impact)

        # El peor caso entre grupos ya no es vacío: los impactos difieren.
        descriptor_disparities[j] = compute_descriptor_disparity_from_group_impact(
            descriptor_group_impact=group_impact,
        )

    return descriptor_disparities, descriptor_group_impacts, baseline_group_disparities


# =========================================================
# REGULARIZACIÓN
# =========================================================

def regularize_mad_directional(
    mad: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    classes: np.ndarray,
    alpha: float = 0.5,
    operator: str = "product",
    thresholds: FairnessThresholds | None = None,
    weights: FairnessWeights | None = None,
    eta: float = 1.0,
    tau_j: float = 0.0,
    min_group_size: int = 1,
    min_positive_size: int = 1,
    default_for_small_groups: float = 0.0,
    class_aggregation: str = "mean",
    clip: bool = True,
    return_details: bool = False,
) -> np.ndarray | dict[str, Any]:
    """
    Regulariza el tensor MAD empleando la señal de control direccional.

    Mantiene la interfaz y el diccionario de resultados de la versión simétrica,
    de modo que puede sustituirla sin más cambios en el clasificador.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor MAD del modelo completo.
    y_true : np.ndarray
        Etiquetas reales.
    sensitive : np.ndarray
        Vector de grupos sensibles.
    classes : np.ndarray
        Vector de etiquetas de clase del modelo.
    alpha : float, default=0.5
        Parámetro de exigencia de LAMDA.
    operator : str, default="product"
        Operador de agregación empleado en el grado de adecuación global.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    eta : float, default=1.0
        Intensidad de la penalización por descriptor.
    tau_j : float, default=0.0
        Umbral de activación por descriptor.
    min_group_size : int, default=1
        Soporte mínimo del grupo.
    min_positive_size : int, default=1
        Soporte mínimo por clase para la diferencia de igualdad de oportunidad.
    default_for_small_groups : float, default=0.0
        Valor asignado a grupos con soporte insuficiente.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.
    clip : bool, default=True
        Si es True, recorta los valores al intervalo numérico válido.
    return_details : bool, default=False
        Si es True, devuelve también los resultados intermedios.

    Retorna
    -------
    np.ndarray o dict[str, Any]
        Tensor MAD regularizado, o diccionario con los resultados intermedios.
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)
    validate_operator(operator)

    descriptor_disparities, descriptor_group_impacts, baseline_group_disparities = (
        compute_all_descriptor_disparities_directional(
            mad=mad,
            y_true=y_true,
            sensitive=sensitive,
            classes=classes,
            alpha=alpha,
            operator=operator,
            thresholds=thresholds,
            weights=weights,
            min_group_size=min_group_size,
            min_positive_size=min_positive_size,
            default_for_small_groups=default_for_small_groups,
            class_aggregation=class_aggregation,
        )
    )

    descriptor_weights = compute_descriptor_weights(
        descriptor_disparities=descriptor_disparities,
        eta=eta,
        tau_j=tau_j,
    )

    mad_fair = apply_mad_fairness(
        mad=mad,
        descriptor_weights=descriptor_weights,
        clip=clip,
    )

    if not return_details:
        return mad_fair

    gad_base = aggregate_mad_to_gad(mad=mad, alpha=alpha, operator=operator)
    gad_fair = aggregate_mad_to_gad(mad=mad_fair, alpha=alpha, operator=operator)

    return {
        "mad_fair": mad_fair,
        "gad_base": gad_base,
        "gad_fair": gad_fair,
        "y_pred_base": predict_class_labels(gad_base, classes),
        "y_pred_fair": predict_class_labels(gad_fair, classes),
        "descriptor_disparities": descriptor_disparities,
        "descriptor_weights": descriptor_weights,
        "descriptor_group_impacts": descriptor_group_impacts,
        "baseline_group_disparities": baseline_group_disparities,
    }
