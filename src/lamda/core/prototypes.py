"""
Cálculo de prototipos de clase para LAMDA clásico.

Este módulo implementa el cálculo del valor prototípico rho_{c,j} de
cada descriptor j en cada clase c. En la formulación clásica de LAMDA,
dicho prototipo se define como la media del descriptor normalizado en
la clase correspondiente.

Formalmente:

    rho_{c,j} = (1 / n_c) * sum_{t=1}^{n_c} x̄_{t,j}

donde n_c es el número de individuos de la clase c.
"""

from __future__ import annotations

import numpy as np

from lamda.utils.numerical import ensure_2d


def validate_X_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida y homogeneiza la matriz de entrada X y el vector de etiquetas y.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de entrada de forma (n_individuos, n_descriptores).
    y : np.ndarray
        Vector de etiquetas de longitud n_individuos.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Tupla (X, y) validada y convertida a arrays de NumPy.

    Lanza
    ------
    ValueError
        Si el número de individuos en X no coincide con la longitud de y.
    """
    X = ensure_2d(np.asarray(X, dtype=float))
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "El número de individuos en X debe coincidir con el número "
            "de etiquetas en y."
        )

    if X.shape[0] == 0:
        raise ValueError("La matriz X debe contener al menos un individuo.")

    return X, y


def compute_class_prototypes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula los prototipos de clase rho_{c,j} para LAMDA clásico.

    Parámetros
    ----------
    X : np.ndarray
        Matriz normalizada de forma (n_individuos, n_descriptores).
    y : np.ndarray
        Vector de etiquetas de clase de longitud n_individuos.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Una tupla formada por:

        - classes_ : np.ndarray
            Vector ordenado con las clases observadas.
        - prototypes : np.ndarray
            Matriz de prototipos de forma (n_clases, n_descriptores),
            donde cada fila corresponde al vector rho_c de una clase.

    Notas
    -----
    Cada prototipo se calcula como la media de los descriptores de los
    individuos pertenecientes a una clase dada. Esta implementación
    asume que X ya ha sido previamente normalizada.
    """
    X, y = validate_X_y(X, y)

    classes_ = np.unique(y)
    n_classes = classes_.shape[0]
    n_features = X.shape[1]

    prototypes = np.zeros((n_classes, n_features), dtype=float)

    for class_index, class_label in enumerate(classes_):
        class_mask = y == class_label
        X_class = X[class_mask]

        # El prototipo rho_c se define como la media descriptor a descriptor
        # de los individuos pertenecientes a la clase c.
        prototypes[class_index, :] = np.mean(X_class, axis=0)

    return classes_, prototypes