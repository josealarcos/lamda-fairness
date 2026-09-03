"""
Regla de decisión para LAMDA clásico.

Este módulo implementa la asignación final de cada individuo a una clase
a partir de la matriz de grados de adecuación global (GAD).

En la versión más simple de LAMDA clásico, cada individuo se asigna a la
clase cuyo GAD es máximo. Más adelante, esta lógica podrá extenderse
para incluir explícitamente la comparación con la clase no informativa
(NIC).
"""

from __future__ import annotations

import numpy as np


def validate_gad_matrix(gad: np.ndarray) -> np.ndarray:
    """
    Valida la matriz de GAD.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.

    Retorna
    -------
    np.ndarray
        Matriz GAD convertida a tipo float.

    Lanza
    ------
    ValueError
        Si la entrada no tiene exactamente dos dimensiones.
    """
    gad = np.asarray(gad, dtype=float)

    if gad.ndim != 2:
        raise ValueError(
            "La matriz GAD debe tener forma (n_individuos, n_clases)."
        )

    if gad.shape[1] == 0:
        raise ValueError(
            "La matriz GAD debe contener al menos una clase."
        )

    return gad


def predict_class_indices(gad: np.ndarray) -> np.ndarray:
    """
    Obtiene el índice de la clase asignada a cada individuo.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.

    Retorna
    -------
    np.ndarray
        Vector de forma (n_individuos,) con los índices de clase
        seleccionados según el máximo GAD por fila.

    Notas
    -----
    Esta función devuelve índices posicionales, no etiquetas originales
    de clase. La conversión de índice a etiqueta se realizará en el
    modelo de alto nivel (`LamdaClassifier`), usando el vector `classes_`.
    """
    gad = validate_gad_matrix(gad)
    return np.argmax(gad, axis=1)


def predict_class_labels(gad: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """
    Obtiene la etiqueta de clase asignada a cada individuo.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.
    classes_ : np.ndarray
        Vector ordenado con las etiquetas reales de clase.

    Retorna
    -------
    np.ndarray
        Vector con las etiquetas predichas para cada individuo.

    Lanza
    ------
    ValueError
        Si el número de clases en `gad` no coincide con la longitud
        de `classes_`.
    """
    gad = validate_gad_matrix(gad)
    classes_ = np.asarray(classes_)

    if gad.shape[1] != classes_.shape[0]:
        raise ValueError(
            "El número de columnas de GAD debe coincidir con el número "
            "de clases disponibles en `classes_`."
        )

    class_indices = predict_class_indices(gad)
    return classes_[class_indices]


def predict_max_gad(gad: np.ndarray) -> np.ndarray:
    """
    Devuelve el valor máximo de GAD para cada individuo.

    Parámetros
    ----------
    gad : np.ndarray
        Matriz de forma (n_individuos, n_clases) con los valores GAD.

    Retorna
    -------
    np.ndarray
        Vector de forma (n_individuos,) con el máximo GAD por individuo.

    Notas
    -----
    Esta función es útil para inspección, depuración y futuras
    extensiones donde la decisión pueda depender no solo de la clase
    ganadora, sino también de la magnitud del GAD máximo.
    """
    gad = validate_gad_matrix(gad)
    return np.max(gad, axis=1)