"""
Implementación del MAD (Marginal Adequacy Degree) para LAMDA clásico.

Este módulo implementa el cálculo del grado de adecuación marginal
(MAD) para cada descriptor j de un individuo r respecto a una clase c.

En la formulación clásica de LAMDA (modelo binomial), el MAD mide el
grado de similitud entre el valor normalizado del descriptor x̄_{r,j}
y el prototipo de clase ρ_{c,j}.

Formalmente, el MAD binomial se basa en una función de tipo:

    MAD_{c,j}(x̄_{r,j} | ρ_{c,j})

que devuelve un valor en el intervalo [0, 1], donde valores cercanos a 1
indican alta adecuación.
"""

from __future__ import annotations

import numpy as np

from lamda.utils.numerical import EPSILON, clip01, ensure_2d


def mad_binomial(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """
    Calcula el MAD binomial para todos los individuos y clases.

    Parámetros
    ----------
    X : np.ndarray
        Matriz normalizada de forma (n_individuos, n_descriptores).
    prototypes : np.ndarray
        Matriz de prototipos de forma (n_clases, n_descriptores).

    Retorna
    -------
    np.ndarray
        Tensor de forma (n_individuos, n_clases, n_descriptores)
        con los valores de MAD.

    Notas
    -----
    - Se aplica clipping a los valores para evitar problemas numéricos
      cuando aparecen 0 o 1 exactos.
    - Esta implementación es vectorizada para mejorar rendimiento.
    """
    X = ensure_2d(np.asarray(X, dtype=float))
    prototypes = ensure_2d(np.asarray(prototypes, dtype=float))

    n_samples, n_features = X.shape
    n_classes = prototypes.shape[0]

    if prototypes.shape[1] != n_features:
        raise ValueError(
            "El número de descriptores en prototypes no coincide con X."
        )

    # Expandimos dimensiones para broadcasting:
    # X -> (n_samples, 1, n_features)
    # prototypes -> (1, n_classes, n_features)
    X_exp = X[:, np.newaxis, :]
    P_exp = prototypes[np.newaxis, :, :]

    # Clipping para estabilidad numérica
    X_clipped = clip01(X_exp, epsilon=EPSILON)
    P_clipped = clip01(P_exp, epsilon=EPSILON)

    # Implementación del MAD binomial de LAMDA:
    # MAD_{c,j}(x̄_{r,j} | ρ_{c,j}) = ρ_{c,j}^{x̄_{r,j}} (1 - ρ_{c,j})^{1 - x̄_{r,j}}
    mad = (P_clipped ** X_clipped) * ((1.0 - P_clipped) ** (1.0 - X_clipped))

    return mad


def mad_single(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """
    Calcula el MAD para un único individuo respecto a todas las clases.

    Parámetros
    ----------
    x : np.ndarray
        Vector de descriptores de forma (n_descriptores,).
    prototypes : np.ndarray
        Matriz de prototipos de forma (n_clases, n_descriptores).

    Retorna
    -------
    np.ndarray
        Matriz de forma (n_clases, n_descriptores) con los valores MAD.
    """
    x = ensure_2d(np.asarray(x, dtype=float))
    return mad_binomial(x, prototypes)[0]