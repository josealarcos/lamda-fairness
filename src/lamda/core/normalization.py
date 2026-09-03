
"""
Funciones de normalización para LAMDA clásico.

Este módulo implementa la transformación de cada descriptor al
intervalo [0, 1], siguiendo la formulación matemática utilizada en
LAMDA clásico:

    x̄_{r,j} = (x_{r,j} - x_{j,min}) / (x_{j,max} - x_{j,min})

La normalización se implementa como una clase sencilla, inspirada en el
estilo de uso de scikit-learn, para facilitar su reutilización durante
las fases de ajuste (``fit``) y transformación (``transform``) del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lamda.utils.numerical import EPSILON, ensure_2d, safe_divide


@dataclass
class LamdaMinMaxNormalizer:
    """
    Normalizador min-max propio para LAMDA.

    Esta clase ajusta, para cada descriptor j, los valores mínimo y
    máximo observados en el conjunto de entrenamiento. Posteriormente,
    aplica la transformación al intervalo [0, 1] usando dichos valores.

    Atributos
    ---------
    data_min_ : np.ndarray | None
        Vector con el mínimo observado para cada descriptor.
    data_max_ : np.ndarray | None
        Vector con el máximo observado para cada descriptor.
    data_range_ : np.ndarray | None
        Vector con el rango observado para cada descriptor.
    n_features_in_ : int | None
        Número de descriptores del conjunto usado en ``fit``.
    is_fitted_ : bool
        Indica si el normalizador ya ha sido ajustado.

    Notas
    -----
    - Si un descriptor es constante, su rango será cero. En ese caso,
      la división se estabiliza mediante ``safe_divide`` para evitar
      errores numéricos.
    - El objetivo principal de esta clase es respetar la formulación de
      LAMDA, no sustituir a los escaladores generales de otras librerías.
    """

    data_min_: np.ndarray | None = field(default=None, init=False)
    data_max_: np.ndarray | None = field(default=None, init=False)
    data_range_: np.ndarray | None = field(default=None, init=False)
    n_features_in_: int | None = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, X: np.ndarray) -> "LamdaMinMaxNormalizer":
        """
        Ajusta el normalizador a partir de una matriz de entrada.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        LamdaMinMaxNormalizer
            La propia instancia ajustada.

        Lanza
        ------
        ValueError
            Si la matriz no contiene al menos un descriptor.
        """
        X = ensure_2d(np.asarray(X, dtype=float))

        if X.shape[1] == 0:
            raise ValueError(
                "La matriz de entrada debe contener al menos un descriptor."
            )

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normaliza una matriz usando los parámetros aprendidos en ``fit``.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Matriz normalizada en el intervalo [0, 1].

        Lanza
        ------
        ValueError
            Si el normalizador no ha sido ajustado o si el número de
            descriptores no coincide con el usado en ``fit``.
        """
        self._check_is_fitted()

        X = ensure_2d(np.asarray(X, dtype=float))

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "El número de descriptores de la entrada no coincide con el "
                "utilizado durante el ajuste del normalizador."
            )

        numerator = X - self.data_min_
        X_normalized = safe_divide(numerator, self.data_range_, epsilon=EPSILON)

        # Por coherencia con la formulación de LAMDA, el resultado debe
        # permanecer dentro del intervalo [0, 1]. Un pequeño clipping
        # evita desviaciones numéricas residuales.
        return np.clip(X_normalized, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta el normalizador y transforma la matriz de entrada.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Matriz normalizada en el intervalo [0, 1].
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Reconstruye la escala original a partir de una matriz normalizada.

        Parámetros
        ----------
        X_normalized : np.ndarray
            Matriz normalizada de forma
            (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Matriz reconstruida en la escala original.

        Lanza
        ------
        ValueError
            Si el normalizador no ha sido ajustado o si el número de
            descriptores no coincide con el usado en ``fit``.
        """
        self._check_is_fitted()

        X_normalized = ensure_2d(np.asarray(X_normalized, dtype=float))

        if X_normalized.shape[1] != self.n_features_in_:
            raise ValueError(
                "El número de descriptores de la entrada no coincide con el "
                "utilizado durante el ajuste del normalizador."
            )

        return X_normalized * self.data_range_ + self.data_min_

    def _check_is_fitted(self) -> None:
        """
        Verifica que el normalizador haya sido ajustado previamente.

        Lanza
        ------
        ValueError
            Si la instancia todavía no ha ejecutado ``fit``.
        """
        if not self.is_fitted_:
            raise ValueError(
                "El normalizador no está ajustado. Ejecuta ``fit`` antes de "
                "llamar a ``transform``, ``fit_transform`` o ``inverse_transform``."
            )
