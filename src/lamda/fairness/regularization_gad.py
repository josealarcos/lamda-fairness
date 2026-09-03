"""
Regularización fairness sobre GAD.

Este módulo implementa un mecanismo de corrección de equidad basado en la
penalización selectiva de la clase preliminarmente asignada a cada individuo.

Formulación
-----------
Sea GAD_{c,r} el grado de adecuación global del individuo r respecto a la clase c.

1. Clase preliminar:
    c_hat(r) = argmax_c GAD_{c,r}

2. Disparidad del grupo:
    D_{g(r)} mide la violación de equidad del grupo sensible al que pertenece r.

3. Regularización:
    GAD_fair_{c,r} = GAD_{c,r} - λ * D_{g(r)} * 1[c = c_hat(r)]

donde:
- λ ≥ 0 controla la intensidad de la penalización
- 1[·] es la función indicadora

4. Decisión final:
    c_hat_fair(r) = argmax_c GAD_fair_{c,r}

Flujo
-----
1. Cálculo de GAD (modelo LAMDA base)
2. Obtención de clase preliminar
3. Asignación de disparidad por individuo
4. Aplicación de penalización selectiva
5. Obtención de predicción final
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lamda.utils.numerical import clip01


# =========================================================
# VALIDACIÓN
# =========================================================

def validate_regularization_inputs(
    gad: np.ndarray,
    sensitive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida las entradas del proceso de regularización.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz de grados de adecuación global de forma (n_individuos, n_clases).
    sensitive : np.ndarray
        Vector de grupos sensibles de longitud n_individuos.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Matriz GAD y vector de grupos sensibles validados.
    """

    gad = np.asarray(gad, dtype=float)
    sensitive = np.asarray(sensitive)

    if gad.ndim != 2:
        raise ValueError("GAD debe ser una matriz 2D.")

    if sensitive.ndim != 1:
        sensitive = sensitive.reshape(-1)

    if gad.shape[0] != sensitive.shape[0]:
        raise ValueError(
            "El número de individuos en GAD debe coincidir con la longitud de sensitive."
        )

    return gad, sensitive


def validate_lambda(lambda_: float) -> None:
    """
    Valida el parámetro de regularización λ.

    Parámetros
    ----------
    lambda_ : float
        Intensidad de la penalización.

    Lanza
    ------
    ValueError
        Si λ es negativo.
    """
    if lambda_ < 0.0:
        raise ValueError("lambda_ debe ser mayor o igual que 0.")


# =========================================================
# DISPARIDAD
# =========================================================

def compute_individual_disparity(
    sensitive: np.ndarray,
    group_disparities: dict[Any, float],
    default_value: float = 0.0,
) -> np.ndarray:
    """
    Asigna a cada individuo la disparidad de su grupo.

    Parámetros
    ----------
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group_disparities : dict[Any, float]
        Disparidad por grupo.
    default_value : float, default=0.0
        Valor asignado si el grupo no está presente.

    Retorna
    -------
    np.ndarray
        Vector de disparidades por individuo.
    """

    sensitive = np.asarray(sensitive)

    return np.array(
        [group_disparities.get(g, default_value) for g in sensitive],
        dtype=float,
    )


# =========================================================
# PREDICCIÓN
# =========================================================

def preliminary_predictions_from_gad(gad: np.ndarray) -> np.ndarray:
    """
    Obtiene la clase preliminar mediante argmax sobre GAD.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz GAD de forma (n_individuos, n_clases).

    Retorna
    -------
    np.ndarray
        Índices de clase preliminar.
    """

    gad = np.asarray(gad, dtype=float)

    if gad.ndim != 2:
        raise ValueError("GAD debe ser 2D.")

    return np.argmax(gad, axis=1).astype(int)


def final_predictions_from_fair_gad(gad_fair: np.ndarray) -> np.ndarray:
    """
    Obtiene la clase final tras regularización.

    Parámetros
    ----------
    gad_fair : np.ndarray
        Matriz GAD regularizada.

    Retorna
    -------
    np.ndarray
        Índices de clase final.
    """

    gad_fair = np.asarray(gad_fair, dtype=float)

    if gad_fair.ndim != 2:
        raise ValueError("gad_fair debe ser 2D.")

    return np.argmax(gad_fair, axis=1).astype(int)


# =========================================================
# REGULARIZACIÓN
# =========================================================

def apply_selective_gad_penalty(
    gad: np.ndarray,
    preliminary_preds: np.ndarray,
    individual_disparity: np.ndarray,
    lambda_: float = 0.1,
    clip: bool = True,
) -> np.ndarray:
    """
    Aplica penalización selectiva sobre la clase preliminar.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz GAD original.
    preliminary_preds : np.ndarray
        Clase preliminar por individuo.
    individual_disparity : np.ndarray
        Disparidad por individuo.
    lambda_ : float, default=0.1
        Intensidad de penalización.
    clip : bool, default=True
        Si True, recorta valores a [0,1].

    Retorna
    -------
    np.ndarray
        Matriz GAD regularizada.
    """

    gad = np.asarray(gad, dtype=float)
    preliminary_preds = np.asarray(preliminary_preds, dtype=int)
    individual_disparity = np.asarray(individual_disparity, dtype=float)

    gad_fair = gad.copy()

    row_idx = np.arange(gad.shape[0])
    gad_fair[row_idx, preliminary_preds] -= lambda_ * individual_disparity

    if clip:
        gad_fair = clip01(gad_fair)

    return gad_fair


# =========================================================
# INTERFAZ PRINCIPAL
# =========================================================

def regularize_gad_from_disparities(
    gad: np.ndarray,
    sensitive: np.ndarray,
    group_disparities: dict[Any, float],
    lambda_: float = 0.1,
    clip: bool = True,
    default_group_disparity: float = 0.0,
    return_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regulariza GAD a partir de disparidades de grupo ya calculadas.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz GAD de forma (n_individuos, n_clases).
    sensitive : np.ndarray
        Vector de grupos sensibles.
    group_disparities : dict[Any, float]
        Diccionario {grupo: D_g}.
    lambda_ : float, default=0.1
        Intensidad de la penalización.
    clip : bool, default=True
        Si True, recorta los valores finales al intervalo [0, 1].
    default_group_disparity : float, default=0.0
        Disparidad asignada a grupos no presentes en el diccionario.
    return_predictions : bool, default=False
        Si True, devuelve también predicciones preliminares y finales.

    Retorna
    -------
    np.ndarray
        Matriz GAD regularizada.

    o bien

    tuple[np.ndarray, np.ndarray, np.ndarray]
        (gad_fair, preliminary_preds, final_preds)
    """
    gad, sensitive = validate_regularization_inputs(gad, sensitive)
    validate_lambda(lambda_)

    individual_disp = compute_individual_disparity(
        sensitive=sensitive,
        group_disparities=group_disparities,
        default_value=default_group_disparity,
    )

    preliminary_preds = preliminary_predictions_from_gad(gad)

    gad_fair = apply_selective_gad_penalty(
        gad=gad,
        preliminary_preds=preliminary_preds,
        individual_disparity=individual_disp,
        lambda_=lambda_,
        clip=clip,
    )

    if not return_predictions:
        return gad_fair

    final_preds = final_predictions_from_fair_gad(gad_fair)
    return gad_fair, preliminary_preds, final_preds