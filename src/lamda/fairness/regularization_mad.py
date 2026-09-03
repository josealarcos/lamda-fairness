"""
Regularización fairness sobre MAD.

Este módulo implementa la propuesta de mitigación a nivel de descriptor,
basada en identificar qué variables contribuyen más al incremento de la
disparidad entre grupos sensibles y reducir progresivamente su influencia
en el cálculo de MAD.

Formulación
-----------
Sea MAD_{c,j}(X_r) el grado de adecuación marginal del descriptor j del individuo r
respecto a la clase c.

1. Disparidad por grupo
   La disparidad por grupo se calcula mediante métricas de fairness multiclase
   en esquema one-vs-rest, agregadas por clase.

2. Impacto del descriptor
   Para cada descriptor j y grupo g se define:

       C_{j,g}^{(t)} = D_g^{(t)} - D_{g,(-j)}^{(t)}

   donde:
   - D_g^{(t)} es la disparidad del grupo g con el modelo completo.
   - D_{g,(-j)}^{(t)} es la disparidad del grupo g cuando el descriptor j
     no participa en la agregación.

3. Disparidad por descriptor
   Se considera el peor caso entre grupos:

       C_j^{(t)} = max_g C_{j,g}^{(t)}

   y la disparidad efectiva del descriptor se define como:

       D_j^{(t)} = max(0, C_j^{(t)})

4. Peso fairness
   A partir de la disparidad del descriptor se define:

       w_j^{(t)} = 1 / (1 + eta * max(0, D_j^{(t)} - tau_j))

   donde:
   - eta >= 0 controla la intensidad de la penalización.
   - tau_j es el umbral de disparidad del descriptor.

5. Regularización de MAD
   El grado de adecuación marginal corregido se define como:

       MAD_{c,j}^{fair,(t)}(X_r) = (MAD_{c,j}(X_r))^{w_j^{(t)}}

   Cuando w_j^{(t)} < 1, el valor de MAD tiende hacia 1, reduciendo la
   capacidad discriminativa del descriptor y neutralizando progresivamente
   su influencia en la decisión final.

Flujo
-----
1. Se calcula la disparidad del modelo completo.
2. Para cada descriptor:
   - se elimina de forma temporal de la agregación,
   - se recalculan predicciones,
   - se estima la nueva disparidad por grupo,
   - se mide el impacto del descriptor.
3. Se construyen pesos fairness por descriptor.
4. Se aplica la transformación sobre el tensor MAD.

Nota de implementación
----------------------
Este módulo regulariza únicamente MAD. No aplica regularización directa sobre GAD.

Sin embargo, para calcular predicciones y medir disparidad es necesario transformar:

    MAD -> GAD -> clase predicha

Para ello se utiliza `compute_gad` de `lamda.core.gad`, la misma función utilizada
por LAMDA clásico. De este modo se garantiza que la propuesta Fair-MAD use los
mismos operadores de agregación que el resto del proyecto:

    - "product"
    - "minmax"
    - "lukasiewicz"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lamda.core.decision import predict_class_labels
from lamda.core.gad import compute_gad
from lamda.fairness.disparity import (
    FairnessThresholds,
    FairnessWeights,
    all_group_disparities,
)
from lamda.utils.numerical import clip01


# =========================================================
# VALIDACIÓN
# =========================================================

def validate_mad_tensor(mad: np.ndarray) -> np.ndarray:
    """
    Valida el tensor MAD.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor de grados de adecuación marginal de forma
        (n_individuos, n_clases, n_descriptores).

    Retorna
    -------
    np.ndarray
        Tensor MAD validado.

    Lanza
    ------
    ValueError
        Si el tensor no tiene tres dimensiones.
    """
    mad = np.asarray(mad, dtype=float)

    if mad.ndim != 3:
        raise ValueError(
            "MAD debe ser un tensor 3D de forma "
            "(n_individuos, n_clases, n_descriptores)."
        )

    return mad


def validate_alpha(alpha: float) -> None:
    """
    Valida el parámetro alpha.

    Parámetros
    ----------
    alpha : float
        Parámetro de exigencia de LAMDA.

    Lanza
    ------
    ValueError
        Si alpha no pertenece al intervalo [0, 1].
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha debe estar en el intervalo [0, 1].")


def validate_operator(operator: str) -> None:
    """
    Valida el operador de agregación GAD.

    Parámetros
    ----------
    operator : str
        Operador de agregación.

    Lanza
    ------
    ValueError
        Si el operador no está soportado.
    """
    valid_operators = {"product", "minmax", "lukasiewicz"}

    if operator not in valid_operators:
        raise ValueError(
            f"Operador no soportado: {operator}. "
            f"Valores admitidos: {sorted(valid_operators)}"
        )


def validate_eta(eta: float) -> None:
    """
    Valida el parámetro eta.

    Parámetros
    ----------
    eta : float
        Intensidad de la penalización.

    Lanza
    ------
    ValueError
        Si eta es negativo.
    """
    if eta < 0.0:
        raise ValueError("eta debe ser mayor o igual que 0.")


def validate_tau_j(tau_j: float) -> None:
    """
    Valida el umbral de disparidad por descriptor.

    Parámetros
    ----------
    tau_j : float
        Umbral de disparidad del descriptor.

    Lanza
    ------
    ValueError
        Si tau_j es negativo.
    """
    if tau_j < 0.0:
        raise ValueError("tau_j debe ser mayor o igual que 0.")


# =========================================================
# UTILIDADES DE AGREGACIÓN
# =========================================================

def aggregate_mad_to_gad(
    mad: np.ndarray,
    alpha: float = 0.5,
    operator: str = "product",
) -> np.ndarray:
    """
    Agrega el tensor MAD para obtener la matriz GAD.

    Esta función no aplica regularización GAD. Su única finalidad es
    transformar MAD -> GAD usando la formulación estándar de LAMDA,
    para poder obtener predicciones y calcular disparidad.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor MAD de forma (n_individuos, n_clases, n_descriptores).
    alpha : float, default=0.5
        Parámetro de exigencia de LAMDA.
    operator : str, default="product"
        Operador de agregación GAD. Valores admitidos:
        - "product"
        - "minmax"
        - "lukasiewicz"

    Retorna
    -------
    np.ndarray
        Matriz GAD de forma (n_individuos, n_clases).
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)
    validate_operator(operator)

    return compute_gad(
        mad=mad,
        alpha=alpha,
        operator=operator,
    )


def predictions_from_mad(
    mad: np.ndarray,
    classes: np.ndarray,
    alpha: float = 0.5,
    operator: str = "product",
) -> np.ndarray:
    """
    Obtiene predicciones finales a partir del tensor MAD.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor MAD de forma (n_individuos, n_clases, n_descriptores).
    classes : np.ndarray
        Vector de etiquetas de clase del modelo.
    alpha : float, default=0.5
        Parámetro de exigencia de LAMDA.
    operator : str, default="product"
        Operador de agregación GAD.

    Retorna
    -------
    np.ndarray
        Vector de etiquetas predichas.
    """
    gad = aggregate_mad_to_gad(
        mad=mad,
        alpha=alpha,
        operator=operator,
    )

    return predict_class_labels(gad, classes)


def remove_descriptor_from_mad(
    mad: np.ndarray,
    descriptor_index: int,
) -> np.ndarray:
    """
    Elimina un descriptor del tensor MAD.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor MAD de forma (n_individuos, n_clases, n_descriptores).
    descriptor_index : int
        Índice del descriptor a eliminar.

    Retorna
    -------
    np.ndarray
        Tensor MAD sin el descriptor indicado.

    Lanza
    ------
    ValueError
        Si el índice está fuera de rango.
    """
    mad = validate_mad_tensor(mad)
    n_descriptors = mad.shape[2]

    if descriptor_index < 0 or descriptor_index >= n_descriptors:
        raise ValueError("descriptor_index fuera de rango.")

    return np.delete(mad, descriptor_index, axis=2)


# =========================================================
# DISPARIDAD DEL MODELO
# =========================================================

def compute_model_group_disparities(
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
    Calcula la disparidad por grupo de un modelo ya evaluado.

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
        Soporte mínimo para evaluar un grupo.
    min_positive_size : int, default=1
        Soporte mínimo por clase para el cálculo de EOD.
    default_for_small_groups : float, default=0.0
        Valor asignado a grupos con soporte insuficiente.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.

    Retorna
    -------
    dict[Any, float]
        Diccionario {grupo: D_g}.
    """
    return all_group_disparities(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        thresholds=thresholds,
        weights=weights,
        min_group_size=min_group_size,
        min_positive_size=min_positive_size,
        default_for_small_groups=default_for_small_groups,
        class_aggregation=class_aggregation,
    )


# =========================================================
# IMPACTO POR DESCRIPTOR
# =========================================================

def compute_descriptor_group_impact(
    baseline_group_disparities: dict[Any, float],
    reduced_group_disparities: dict[Any, float],
) -> dict[Any, float]:
    """
    Calcula el impacto de un descriptor sobre la disparidad de cada grupo.

    Parámetros
    ----------
    baseline_group_disparities : dict[Any, float]
        Disparidad por grupo con el modelo completo.
    reduced_group_disparities : dict[Any, float]
        Disparidad por grupo con el descriptor eliminado.

    Retorna
    -------
    dict[Any, float]
        Diccionario {grupo: C_{j,g}} con el impacto del descriptor.
    """
    groups = sorted(
        set(baseline_group_disparities.keys()) | set(reduced_group_disparities.keys())
    )

    impact: dict[Any, float] = {}

    for group in groups:
        baseline = baseline_group_disparities.get(group, 0.0)
        reduced = reduced_group_disparities.get(group, 0.0)
        impact[group] = float(baseline - reduced)

    return impact


def compute_descriptor_disparity_from_group_impact(
    descriptor_group_impact: dict[Any, float],
) -> float:
    """
    Calcula la disparidad efectiva de un descriptor a partir del impacto por grupo.

    Parámetros
    ----------
    descriptor_group_impact : dict[Any, float]
        Diccionario {grupo: C_{j,g}}.

    Retorna
    -------
    float
        Disparidad efectiva del descriptor D_j.
    """
    if len(descriptor_group_impact) == 0:
        return 0.0

    worst_case = max(descriptor_group_impact.values())
    return float(max(0.0, worst_case))


def compute_all_descriptor_disparities(
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
    Calcula la disparidad efectiva de todos los descriptores.

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
        Operador de agregación utilizado en GAD.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    min_group_size : int, default=1
        Soporte mínimo para evaluar un grupo.
    min_positive_size : int, default=1
        Soporte mínimo por clase para el cálculo de EOD.
    default_for_small_groups : float, default=0.0
        Valor asignado a grupos con soporte insuficiente.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.

    Retorna
    -------
    tuple[np.ndarray, list[dict[Any, float]], dict[Any, float]]
        - Vector de disparidades por descriptor D_j.
        - Lista con impactos por grupo para cada descriptor.
        - Disparidad por grupo del modelo completo.
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

    baseline_group_disparities = compute_model_group_disparities(
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
        mad_reduced = remove_descriptor_from_mad(
            mad=mad,
            descriptor_index=j,
        )

        reduced_pred = predictions_from_mad(
            mad=mad_reduced,
            classes=classes,
            alpha=alpha,
            operator=operator,
        )

        reduced_group_disparities = compute_model_group_disparities(
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

        descriptor_disparities[j] = compute_descriptor_disparity_from_group_impact(
            descriptor_group_impact=group_impact,
        )

    return descriptor_disparities, descriptor_group_impacts, baseline_group_disparities


# =========================================================
# PESOS FAIRNESS
# =========================================================

def compute_descriptor_weights(
    descriptor_disparities: np.ndarray,
    eta: float = 1.0,
    tau_j: float = 0.0,
) -> np.ndarray:
    """
    Calcula pesos fairness por descriptor.

    Parámetros
    ----------
    descriptor_disparities : np.ndarray
        Vector de disparidades efectivas D_j.
    eta : float, default=1.0
        Intensidad de la penalización.
    tau_j : float, default=0.0
        Umbral de disparidad del descriptor.

    Retorna
    -------
    np.ndarray
        Vector de pesos fairness w_j.
    """
    descriptor_disparities = np.asarray(descriptor_disparities, dtype=float).reshape(-1)

    validate_eta(eta)
    validate_tau_j(tau_j)

    excess = np.maximum(0.0, descriptor_disparities - tau_j)
    descriptor_weights = 1.0 / (1.0 + eta * excess)

    return descriptor_weights.astype(float)


# =========================================================
# REGULARIZACIÓN SOBRE MAD
# =========================================================

def apply_mad_fairness(
    mad: np.ndarray,
    descriptor_weights: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """
    Aplica la regularización fairness sobre el tensor MAD.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor MAD de forma (n_individuos, n_clases, n_descriptores).
    descriptor_weights : np.ndarray
        Vector de pesos fairness de longitud n_descriptores.
    clip : bool, default=True
        Si True, recorta los valores al intervalo numérico válido.

    Retorna
    -------
    np.ndarray
        Tensor MAD regularizado.
    """
    mad = validate_mad_tensor(mad)
    descriptor_weights = np.asarray(descriptor_weights, dtype=float).reshape(-1)

    n_descriptors = mad.shape[2]

    if descriptor_weights.shape[0] != n_descriptors:
        raise ValueError(
            "descriptor_weights debe tener la misma longitud que el número de descriptores."
        )

    mad_fair = np.empty_like(mad)

    for j in range(n_descriptors):
        mad_fair[:, :, j] = np.power(mad[:, :, j], descriptor_weights[j])

    if clip:
        mad_fair = clip01(mad_fair)

    return mad_fair


def regularize_mad(
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
    Regulariza el tensor MAD calculando primero el impacto de cada descriptor.

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
        Operador de agregación utilizado en GAD.
    thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    weights : FairnessWeights | None, default=None
        Pesos de agregación.
    eta : float, default=1.0
        Intensidad de la penalización.
    tau_j : float, default=0.0
        Umbral de disparidad del descriptor.
    min_group_size : int, default=1
        Soporte mínimo para evaluar un grupo.
    min_positive_size : int, default=1
        Soporte mínimo por clase para el cálculo de EOD.
    default_for_small_groups : float, default=0.0
        Valor asignado a grupos con soporte insuficiente.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases.
    clip : bool, default=True
        Si True, recorta los valores al intervalo numérico válido.
    return_details : bool, default=False
        Si True, devuelve también información intermedia.

    Retorna
    -------
    np.ndarray o dict[str, Any]
        Si return_details=False, devuelve el tensor MAD regularizado.
        Si return_details=True, devuelve un diccionario con resultados
        intermedios y finales.
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)
    validate_operator(operator)

    descriptor_disparities, descriptor_group_impacts, baseline_group_disparities = (
        compute_all_descriptor_disparities(
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

    gad_base = aggregate_mad_to_gad(
        mad=mad,
        alpha=alpha,
        operator=operator,
    )

    gad_fair = aggregate_mad_to_gad(
        mad=mad_fair,
        alpha=alpha,
        operator=operator,
    )

    y_pred_base = predict_class_labels(gad_base, classes)
    y_pred_fair = predict_class_labels(gad_fair, classes)

    return {
        "mad_fair": mad_fair,
        "gad_base": gad_base,
        "gad_fair": gad_fair,
        "y_pred_base": y_pred_base,
        "y_pred_fair": y_pred_fair,
        "descriptor_disparities": descriptor_disparities,
        "descriptor_weights": descriptor_weights,
        "descriptor_group_impacts": descriptor_group_impacts,
        "baseline_group_disparities": baseline_group_disparities,
    }