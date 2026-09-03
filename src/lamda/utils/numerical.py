"""
Utilidades numéricas comunes para la implementación de LAMDA.

Este módulo centraliza pequeñas funciones de estabilidad numérica que
se reutilizarán en varias partes del proyecto, especialmente en:

- normalización de descriptores en [0, 1],
- cálculo del MAD binomial,
- operadores de agregación GAD.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Epsilon global del proyecto.
# Se utiliza para evitar divisiones por cero y para recortar valores
# extremos cuando sea necesario.
EPSILON: Final[float] = 1e-12


def clip01(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """
    Recorta un array al intervalo cerrado [epsilon, 1 - epsilon].

    Parámetros
    ----------
    values : np.ndarray
        Array de entrada cuyos valores se desean estabilizar.
    epsilon : float, default=EPSILON
        Margen de seguridad usado para evitar valores exactamente
        iguales a 0 o 1.

    Retorna
    -------
    np.ndarray
        Array recortado al intervalo [epsilon, 1 - epsilon].

    Notas
    -----
    Esta función será especialmente útil en el cálculo del MAD binomial,
    donde valores exactos 0 o 1 pueden generar comportamientos numéricos
    inestables en ciertas operaciones posteriores.
    """
    return np.clip(values, epsilon, 1.0 - epsilon)


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Realiza una división elemento a elemento evitando divisiones por cero.

    Parámetros
    ----------
    numerator : np.ndarray
        Numerador de la división.
    denominator : np.ndarray
        Denominador de la división.
    epsilon : float, default=EPSILON
        Valor mínimo absoluto permitido en el denominador.

    Retorna
    -------
    np.ndarray
        Resultado de dividir `numerator` entre un denominador estabilizado.

    Notas
    -----
    Si algún elemento del denominador tiene magnitud menor que `epsilon`,
    se sustituye por `epsilon` conservando el signo cuando es posible.
    En la práctica del proyecto, esto evita errores numéricos sin alterar
    de forma significativa la lógica del algoritmo.
    """
    stabilized_denominator = np.where(
        np.abs(denominator) < epsilon,
        np.where(denominator < 0, -epsilon, epsilon),
        denominator,
    )
    return numerator / stabilized_denominator


def ensure_2d(array: np.ndarray) -> np.ndarray:
    """
    Garantiza que la entrada tenga dos dimensiones.

    Parámetros
    ----------
    array : np.ndarray
        Array de entrada.

    Retorna
    -------
    np.ndarray
        Array con forma bidimensional.

    Lanza
    -----
    ValueError
        Si el array tiene un número de dimensiones distinto de 1 o 2.

    Notas
    -----
    En LAMDA, la matriz de entrada X debe interpretarse como una matriz
    de forma (n_individuos, n_descriptores). Esta función ayuda a
    homogeneizar entradas antes de operar con ellas.
    """
    if array.ndim == 1:
        return array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(
            "La entrada debe ser un array 1D o 2D para poder interpretarse "
            "como una matriz de individuos y descriptores."
        )

    return array