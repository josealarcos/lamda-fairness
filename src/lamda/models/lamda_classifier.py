"""
Modelo principal LamdaClassifier para LAMDA clásico.

Este módulo implementa una primera versión funcional de un clasificador
LAMDA con interfaz inspirada en scikit-learn. El flujo interno del
modelo sigue la estructura clásica:

    X -> normalización -> prototipos -> MAD -> GAD -> decisión

En esta primera implementación:
- se usa normalización min-max propia de LAMDA,
- se calculan prototipos por clase,
- se calcula el MAD binomial,
- se agrega mediante GAD,
- y la asignación final se realiza por máximo GAD.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from lamda.core.decision import predict_class_labels
from lamda.core.gad import compute_gad
from lamda.core.mad import mad_binomial
from lamda.core.normalization import LamdaMinMaxNormalizer
from lamda.core.prototypes import compute_class_prototypes, validate_X_y


class LamdaClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador LAMDA clásico.

    Parámetros
    ----------
    alpha : float, default=0.5
        Parámetro de exigencia del modelo. Debe pertenecer al intervalo
        [0, 1].
    operator : str, default="product"
        Operador de agregación GAD. Valores admitidos:
        - "product"
        - "minmax"
        - "lukasiewicz"

    Atributos
    ---------
    classes_ : np.ndarray
        Vector ordenado de etiquetas de clase observadas durante el ajuste.
    prototypes_ : np.ndarray
        Matriz de prototipos de forma (n_clases, n_descriptores).
    normalizer_ : LamdaMinMaxNormalizer
        Normalizador ajustado sobre los datos de entrenamiento.
    n_features_in_ : int
        Número de descriptores del conjunto de entrenamiento.
    is_fitted_ : bool
        Indica si el modelo ya ha sido ajustado.
    """

    def __init__(self, alpha: float = 0.5, operator: str = "product") -> None:
        self.alpha = alpha
        self.operator = operator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LamdaClassifier":
        """
        Ajusta el clasificador LAMDA sobre un conjunto de entrenamiento.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).
        y : np.ndarray
            Vector de etiquetas de clase de longitud n_individuos.

        Retorna
        -------
        LamdaClassifier
            La propia instancia ajustada.

        Notas
        -----
        El ajuste consiste en:
        1. validar X e y,
        2. ajustar el normalizador,
        3. normalizar X,
        4. calcular los prototipos por clase.
        """
        X, y = validate_X_y(X, y)

        self.normalizer_ = LamdaMinMaxNormalizer()
        X_normalized = self.normalizer_.fit_transform(X)

        self.classes_, self.prototypes_ = compute_class_prototypes(X_normalized, y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def predict_mad(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula el tensor de MAD para una matriz de entrada.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Tensor de forma (n_individuos, n_clases, n_descriptores)
            con los valores de MAD.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)
        X_normalized = self.normalizer_.transform(X)

        return mad_binomial(X_normalized, self.prototypes_)

    def predict_gad(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de GAD para una matriz de entrada.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Matriz de forma (n_individuos, n_clases) con los valores GAD.
        """
        self._check_is_fitted()

        mad = self.predict_mad(X)
        gad = compute_gad(mad=mad, alpha=self.alpha, operator=self.operator)

        return gad

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve la matriz de adecuación global del modelo.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Matriz GAD de forma (n_individuos, n_clases).

        Notas
        -----
        Esta función se ofrece como interfaz de inspección del modelo,
        análoga a la idea de una "decision function" en otros clasificadores.
        En LAMDA, estos valores no deben interpretarse directamente como
        probabilidades, sino como grados de adecuación global.
        """
        return self.predict_gad(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase de cada individuo de la matriz de entrada.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).

        Retorna
        -------
        np.ndarray
            Vector con las etiquetas de clase predichas.
        """
        self._check_is_fitted()

        gad = self.predict_gad(X)
        return predict_class_labels(gad, self.classes_)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Ajusta el modelo y devuelve las predicciones sobre el mismo conjunto.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada de forma (n_individuos, n_descriptores).
        y : np.ndarray
            Vector de etiquetas de clase.

        Retorna
        -------
        np.ndarray
            Vector de clases predichas.
        """
        return self.fit(X, y).predict(X)

    def _check_is_fitted(self) -> None:
        """
        Verifica que el modelo haya sido ajustado previamente.

        Lanza
        ------
        ValueError
            Si el clasificador todavía no ha sido ajustado.
        """
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError(
                "El modelo LamdaClassifier no está ajustado. "
                "Ejecuta `fit` antes de llamar a `predict`, "
                "`predict_mad`, `predict_gad` o `decision_function`."
            )
