"""
Implementación del GAD (Global Adequacy Degree) para LAMDA clásico.

Este módulo implementa la agregación de los grados de adecuación
marginal (MAD) para obtener el grado de adecuación global (GAD) de
cada individuo respecto a cada clase.

En LAMDA clásico, el GAD se define como una combinación lineal de una
agregación conjuntiva T(·) y una agregación disyuntiva S(·), regulada
por el parámetro de exigencia α:

    GAD_{c,X_r} = α T(MAD_{c,1}, ..., MAD_{c,p})
                  + (1 - α) S(MAD_{c,1}, ..., MAD_{c,p})

En este proyecto se implementan tres variantes de agregación:

- product
- minmax
- lukasiewicz
"""

from __future__ import annotations

import numpy as np

def validate_mad_tensor(mad: np.ndarray) -> np.ndarray:
    """
    Valida el tensor de entrada MAD.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor de forma (n_individuos, n_clases, n_descriptores).

    Retorna
    -------
    np.ndarray
        Tensor MAD convertido a tipo float.

    Lanza
    ------
    ValueError
        Si la entrada no tiene exactamente tres dimensiones.
    """
    mad = np.asarray(mad, dtype=float)

    if mad.ndim != 3:
        raise ValueError(
            "El tensor MAD debe tener forma "
            "(n_individuos, n_clases, n_descriptores)."
        )

    return mad

def gad_product(mad: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calcula el GAD usando la agregación producto / suma probabilística.

    Fórmula
    -------
    GAD^{prod}_{c,X_r} =
        α * Π_j MAD_{c,j}
        + (1 - α) * (1 - Π_j (1 - MAD_{c,j}))

    Parámetros
    ----------
    mad : np.ndarray
        Tensor de forma (n_individuos, n_clases, n_descriptores).
    alpha : float
        Parámetro de exigencia del modelo, con alpha en [0, 1].

    Retorna
    -------
    np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)

    product_term = np.prod(mad, axis=2)
    prob_sum_term = 1.0 - np.prod(1.0 - mad, axis=2)
    gad = alpha * product_term + (1.0 - alpha) * prob_sum_term

    return gad


def gad_minmax(mad: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calcula el GAD usando la agregación min-max.

    Fórmula
    -------
    GAD^{min-max}_{c,X_r} =
        α * min_j(MAD_{c,j}) + (1 - α) * max_j(MAD_{c,j})

    Parámetros
    ----------
    mad : np.ndarray
        Tensor de forma (n_individuos, n_clases, n_descriptores).
    alpha : float
        Parámetro de exigencia del modelo, con alpha en [0, 1].

    Retorna
    -------
    np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)

    min_term = np.min(mad, axis=2)
    max_term = np.max(mad, axis=2)

    gad = alpha * min_term + (1.0 - alpha) * max_term

    return gad


def gad_lukasiewicz(mad: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calcula el GAD usando la agregación de Lukasiewicz.

    Fórmula
    -------
    GAD^{Luk}_{c,X_r} =
        α * max(0, sum_j MAD_{c,j} - (p - 1))
        + (1 - α) * min(1, sum_j MAD_{c,j})

    donde p es el número de descriptores.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor de forma (n_individuos, n_clases, n_descriptores).
    alpha : float
        Parámetro de exigencia del modelo, con alpha en [0, 1].

    Retorna
    -------
    np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.
    """
    mad = validate_mad_tensor(mad)
    validate_alpha(alpha)

    n_features = mad.shape[2]
    sum_term = np.sum(mad, axis=2)

    conjunctive_term = np.maximum(0.0, sum_term - (n_features - 1))
    disjunctive_term = np.minimum(1.0, sum_term)

    gad = alpha * conjunctive_term + (1.0 - alpha) * disjunctive_term

    return gad


def compute_gad(mad: np.ndarray, alpha: float, operator: str = "product") -> np.ndarray:
    """
    Calcula el GAD según el operador de agregación seleccionado.

    Parámetros
    ----------
    mad : np.ndarray
        Tensor de forma (n_individuos, n_clases, n_descriptores).
    alpha : float
        Parámetro de exigencia del modelo, con alpha en [0, 1].
    operator : str, default=\"product\"
        Operador de agregación. Valores admitidos:
        - \"product\"
        - \"minmax\"
        - \"lukasiewicz\"

    Retorna
    -------
    np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.

    Lanza
    ------
    ValueError
        Si el operador indicado no está soportado.
    """
    operator = operator.lower()

    if operator == "product":
        return gad_product(mad, alpha)

    if operator == "minmax":
        return gad_minmax(mad, alpha)

    if operator == "lukasiewicz":
        return gad_lukasiewicz(mad, alpha)

    raise ValueError(
        "Operador GAD no soportado. Usa 'product', 'minmax' o 'lukasiewicz'."
    )


def validate_alpha(alpha: float) -> None:
    """
    Valida que el parámetro alpha pertenezca al intervalo [0, 1].

    Parámetros
    ----------
    alpha : float
        Parámetro de exigencia del modelo.

    Lanza
    ------
    ValueError
        Si alpha no pertenece al intervalo [0, 1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("El parámetro alpha debe pertenecer al intervalo [0, 1].")