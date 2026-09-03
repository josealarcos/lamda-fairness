"""
Modelo LamdaFairGADClassifier.

Este módulo implementa una extensión de LAMDA clásico que incorpora
regularización fairness sobre GAD en un esquema iterativo por bloques.

Formulación
-----------
Sea GAD_{c,r}^{(t)} el grado de adecuación global del individuo r respecto
a la clase c en el bloque t.

1. Clase preliminar:
    c_hat_pre(r, t) = argmax_c GAD_{c,r}^{(t)}

2. Disparidad por grupo:
    D_g^{(t-1)} se calcula a partir del histórico acumulado hasta el bloque
    anterior.

3. Regularización:
    GAD_fair_{c,r}^{(t)} =
        GAD_{c,r}^{(t)} - lambda * D_{g(r)}^{(t-1)} * 1[c = c_hat_pre(r, t)]

4. Decisión final:
    c_hat_fair(r, t) = argmax_c GAD_fair_{c,r}^{(t)}

Flujo
-----
1. Se ajusta el modelo base LAMDA.
2. Se divide el conjunto en bloques de tamaño fijo.
3. Para cada bloque t:
   - se calcula el GAD preliminar,
   - se aplica la penalización usando las disparidades acumuladas hasta t-1,
   - se obtiene la predicción final del bloque,
   - se actualiza el histórico con las etiquetas reales y las predicciones finales,
   - se recalculan las disparidades para el siguiente bloque.
4. El primer bloque se predice sin penalización si no existe histórico previo.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lamda.core.decision import predict_class_labels
from lamda.fairness.directional import (
    all_directional_disparities,
    regularize_gad_directional,
)
from lamda.fairness.disparity import (
    FairnessThresholds,
    FairnessWeights,
    all_group_disparities,
    group_disparity_report,
)
from lamda.fairness.regularization_gad import (
    regularize_gad_from_disparities,
)
from lamda.models.lamda_classifier import LamdaClassifier


class LamdaFairGADClassifier(LamdaClassifier):
    """
    Clasificador LAMDA con regularización fairness sobre GAD.

    Parámetros
    ----------
    alpha : float, default=0.5
        Parámetro de exigencia del modelo base.
    operator : str, default="product"
        Operador de agregación GAD del modelo base.
    lambda_ : float, default=0.1
        Intensidad de la penalización fairness.
    block_size : int, default=128
        Tamaño de bloque usado en la predicción iterativa.
    fairness_thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad. Si es None, se usan los valores por defecto.
    fairness_weights : FairnessWeights | None, default=None
        Pesos de agregación para la disparidad. Si es None, se usan los
        valores por defecto.
    clip : bool, default=True
        Si True, recorta el GAD regularizado al intervalo [0, 1].
    min_group_size : int, default=1
        Tamaño mínimo requerido para evaluar un grupo sensible.
    min_positive_size : int, default=1
        Número mínimo de ejemplos positivos reales requerido para evaluar EOD.
    default_group_disparity : float, default=0.0
        Disparidad asignada a grupos no vistos o no evaluables.
    use_cumulative_history : bool, default=True
        Si True, las disparidades se calculan usando todo el histórico
        acumulado. Si False, se calculan solo con el último bloque.

    Atributos
    ---------
    group_disparities_ : dict[Any, float] | None
        Diccionario con las disparidades vigentes por grupo sensible.
    fairness_is_fitted_ : bool
        Indica si el modelo dispone de disparidades fairness calculadas.
    y_true_history_ : np.ndarray | None
        Histórico acumulado de etiquetas reales.
    y_pred_history_ : np.ndarray | None
        Histórico acumulado de etiquetas predichas finales.
    sensitive_history_ : np.ndarray | None
        Histórico acumulado de grupos sensibles.
    fairness_reports_ : list[dict[Any, dict[str, float]]]
        Informes detallados de fairness por actualización.
    block_history_ : list[dict[str, Any]]
        Historial resumido de bloques procesados.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        operator: str = "product",
        lambda_: float = 0.1,
        block_size: int = 128,
        fairness_thresholds: FairnessThresholds | None = None,
        fairness_weights: FairnessWeights | None = None,
        clip: bool = True,
        min_group_size: int = 1,
        min_positive_size: int = 1,
        default_group_disparity: float = 0.0,
        class_aggregation: str = "mean",
        use_cumulative_history: bool = True,
        disparity_mode: str = "directional",
    ) -> None:
        """
        Inicializa el clasificador fairness sobre GAD.

        Parámetros
        ----------
        alpha : float, default=0.5
            Parámetro de exigencia del modelo base.
        operator : str, default="product"
            Operador de agregación GAD del modelo base.
        lambda_ : float, default=0.1
            Intensidad de la penalización fairness.
        block_size : int, default=128
            Tamaño de bloque usado en la predicción iterativa.
        fairness_thresholds : FairnessThresholds | None, default=None
            Umbrales de equidad.
        fairness_weights : FairnessWeights | None, default=None
            Pesos de agregación.
        clip : bool, default=True
            Si True, recorta el GAD regularizado al intervalo [0, 1].
        min_group_size : int, default=1
            Tamaño mínimo requerido para evaluar un grupo sensible.
        min_positive_size : int, default=1
            Número mínimo de ejemplos positivos reales requerido para evaluar EOD.
        default_group_disparity : float, default=0.0
            Disparidad asignada a grupos no vistos o no evaluables.
        class_aggregation : str, default="mean"
            Método de agregación para calcular las disparidades.
        use_cumulative_history : bool, default=True
            Si True, las disparidades se calculan con histórico acumulado.
        disparity_mode : {"directional", "symmetric"}, default="directional"
            Señal de control empleada en la penalización. En modo direccional se
            emplea la disparidad con signo indexada por par de grupo y clase,
            que solo es positiva cuando el grupo está sobreasignado a la clase.
            En modo simétrico se emplea la disparidad agregada por grupo, que
            con un atributo sensible binario toma el mismo valor en ambos grupos
            y produce por tanto una penalización uniforme. El modo simétrico se
            conserva únicamente para reproducir la formulación inicial en el
            análisis de sensibilidad.
        """
        if disparity_mode not in ("directional", "symmetric"):
            raise ValueError("disparity_mode debe ser 'directional' o 'symmetric'.")

        super().__init__(alpha=alpha, operator=operator)

        self.lambda_ = lambda_
        self.block_size = block_size
        self.fairness_thresholds = fairness_thresholds or FairnessThresholds()
        self.fairness_weights = fairness_weights or FairnessWeights()
        self.clip = clip
        self.min_group_size = min_group_size
        self.min_positive_size = min_positive_size
        self.default_group_disparity = default_group_disparity
        # Estrategia de agregación sobre clases para construir D_g.
        # "mean" (por defecto) penaliza la disparidad promedio del grupo entre
        # clases; se mantiene como valor por defecto por estabilidad de Fair-GAD.
        # El MEO (agregación por máximo) usado para comparar con la literatura
        # multiclase se calcula aparte en las funciones de reporte crudo y no
        # gobierna la penalización.
        self.class_aggregation = class_aggregation
        self.use_cumulative_history = use_cumulative_history
        self.disparity_mode = disparity_mode
        # Señal direccional por par de grupo y clase. Se mantiene junto a la
        # simétrica, que sigue siendo la magnitud de reporte.
        self.directional_disparities_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LamdaFairGADClassifier":
        """
        Ajusta el modelo base LAMDA y reinicia el estado fairness.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada.
        y : np.ndarray
            Vector de etiquetas.

        Retorna
        -------
        LamdaFairGADClassifier
            La propia instancia ajustada.
        """
        super().fit(X, y)

        self.group_disparities_ = None
        self.directional_disparities_ = None
        self.fairness_is_fitted_ = False

        self.y_true_history_ = None
        self.y_pred_history_ = None
        self.sensitive_history_ = None

        self.fairness_reports_ = []
        self.block_history_ = []

        return self

    def update_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> "LamdaFairGADClassifier":
        """
        Actualiza las disparidades fairness del modelo a partir de datos observados.

        Parámetros
        ----------
        y_true : np.ndarray
            Etiquetas reales.
        y_pred : np.ndarray
            Etiquetas predichas finales.
        sensitive : np.ndarray
            Vector de grupos sensibles.

        Retorna
        -------
        LamdaFairGADClassifier
            La propia instancia con el estado fairness actualizado.
        """
        self._check_is_fitted()

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        sensitive = np.asarray(sensitive).reshape(-1)

        if self.use_cumulative_history:
            y_true_ref, y_pred_ref, sensitive_ref = self._append_to_history(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
            )
        else:
            y_true_ref, y_pred_ref, sensitive_ref = y_true, y_pred, sensitive
            self._store_last_block_as_history(
                y_true=y_true,
                y_pred=y_pred,
                sensitive=sensitive,
            )

        self.group_disparities_ = all_group_disparities(
            y_true=y_true_ref,
            y_pred=y_pred_ref,
            sensitive=sensitive_ref,
            thresholds=self.fairness_thresholds,
            weights=self.fairness_weights,
            min_group_size=self.min_group_size,
            min_positive_size=self.min_positive_size,
            default_for_small_groups=self.default_group_disparity,
            class_aggregation=self.class_aggregation,
        )

        # Señal direccional, empleada como control cuando el modo lo indica.
        # La simétrica se calcula siempre, ya que es la magnitud de reporte.
        self.directional_disparities_ = all_directional_disparities(
            y_true=y_true_ref,
            y_pred=y_pred_ref,
            sensitive=sensitive_ref,
            thresholds=self.fairness_thresholds,
            weights=self.fairness_weights,
            min_group_size=self.min_group_size,
            min_positive_size=self.min_positive_size,
            default_for_small_groups=self.default_group_disparity,
        )

        report = group_disparity_report(
            y_true=y_true_ref,
            y_pred=y_pred_ref,
            sensitive=sensitive_ref,
            thresholds=self.fairness_thresholds,
            weights=self.fairness_weights,
            class_aggregation=self.class_aggregation,
        )

        self.fairness_reports_.append(report)
        self.fairness_is_fitted_ = True

        return self

    def predict_gad_fair_from_disparities(
        self,
        X: np.ndarray,
        sensitive: np.ndarray,
        lambda_: float | None = None,
        clip: bool | None = None,
        default_group_disparity: float | None = None,
    ) -> np.ndarray:
        """
        Calcula GAD regularizado a partir de las disparidades almacenadas en el modelo.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada.
        sensitive : np.ndarray
            Vector de grupos sensibles asociado a X.
        lambda_ : float | None, default=None
            Intensidad de la penalización.
        clip : bool | None, default=None
            Si True, recorta el GAD regularizado al intervalo [0, 1].
        default_group_disparity : float | None, default=None
            Disparidad por defecto para grupos no vistos.

        Retorna
        -------
        np.ndarray
            Matriz GAD regularizada.
        """
        self._check_is_fitted()
        self._check_fairness_is_fitted()

        lambda_ = self.lambda_ if lambda_ is None else lambda_
        clip = self.clip if clip is None else clip
        if default_group_disparity is None:
            default_group_disparity = self.default_group_disparity

        gad = self.predict_gad(X)

        if self.disparity_mode == "directional":
            return regularize_gad_directional(
                gad=gad,
                sensitive=sensitive,
                classes=self.classes_,
                disparities=self.directional_disparities_ or {},
                lambda_=lambda_,
                clip=clip,
                default_group_disparity=default_group_disparity,
                only_preliminary_class=True,
            )

        return regularize_gad_from_disparities(
            gad=gad,
            sensitive=sensitive,
            group_disparities=self.group_disparities_,
            lambda_=lambda_,
            clip=clip,
            default_group_disparity=default_group_disparity,
            return_predictions=False,
        )

    def predict_fair_from_disparities(
        self,
        X: np.ndarray,
        sensitive: np.ndarray,
        lambda_: float | None = None,
        clip: bool | None = None,
        default_group_disparity: float | None = None,
    ) -> np.ndarray:
        """
        Predice etiquetas usando las disparidades almacenadas en el modelo.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada.
        sensitive : np.ndarray
            Vector de grupos sensibles asociado a X.
        lambda_ : float | None, default=None
            Intensidad de la penalización.
        clip : bool | None, default=None
            Si True, recorta el GAD regularizado al intervalo [0, 1].
        default_group_disparity : float | None, default=None
            Disparidad por defecto para grupos no vistos.

        Retorna
        -------
        np.ndarray
            Vector con etiquetas predichas.
        """
        gad_fair = self.predict_gad_fair_from_disparities(
            X=X,
            sensitive=sensitive,
            lambda_=lambda_,
            clip=clip,
            default_group_disparity=default_group_disparity,
        )

        return predict_class_labels(gad_fair, self.classes_)

    def predict_blocks(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        sensitive: np.ndarray,
        block_size: int | None = None,
        return_details: bool = False,
    ) -> np.ndarray | dict[str, Any]:
        """
        Ejecuta la predicción fairness en un esquema iterativo por bloques.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada.
        y_true : np.ndarray
            Etiquetas reales asociadas a X.
        sensitive : np.ndarray
            Vector de grupos sensibles asociado a X.
        block_size : int | None, default=None
            Tamaño de bloque. Si es None, se usa el valor del modelo.
        return_details : bool, default=False
            Si True, devuelve también información detallada de cada bloque.

        Retorna
        -------
        np.ndarray o dict[str, Any]
            Si return_details=False, devuelve el vector de predicciones finales.
            Si return_details=True, devuelve un diccionario con resultados
            detallados por bloque.
        """
        self._check_is_fitted()

        X = np.asarray(X)
        y_true = np.asarray(y_true).reshape(-1)
        sensitive = np.asarray(sensitive).reshape(-1)

        if X.shape[0] != y_true.shape[0] or X.shape[0] != sensitive.shape[0]:
            raise ValueError("X, y_true y sensitive deben tener la misma longitud.")

        block_size = self.block_size if block_size is None else block_size
        if block_size <= 0:
            raise ValueError("block_size debe ser mayor que 0.")

        n_samples = X.shape[0]

        final_preds_all = np.empty(n_samples, dtype=self.classes_.dtype)
        preliminary_preds_all = np.empty(n_samples, dtype=self.classes_.dtype)

        gad_fair_blocks: list[np.ndarray] = []
        gad_preliminary_blocks: list[np.ndarray] = []
        block_summaries: list[dict[str, Any]] = []

        for block_index, (start, end) in enumerate(self._block_ranges(n_samples, block_size)):
            X_block = X[start:end]
            y_block = y_true[start:end]
            s_block = sensitive[start:end]

            block_result = self._predict_single_block(
                X_block=X_block,
                sensitive_block=s_block,
            )

            gad_pre_block = block_result["gad_preliminary"]
            gad_fair_block = block_result["gad_fair"]
            prelim_block = block_result["preliminary_preds"]
            final_block = block_result["final_preds"]
            disparities_before = block_result["group_disparities_used"]

            preliminary_preds_all[start:end] = prelim_block
            final_preds_all[start:end] = final_block

            gad_preliminary_blocks.append(gad_pre_block)
            gad_fair_blocks.append(gad_fair_block)

            self.update_fairness(
                y_true=y_block,
                y_pred=final_block,
                sensitive=s_block,
            )

            block_summary = {
                "block_index": block_index,
                "start": start,
                "end": end,
                "n_samples": end - start,
                "group_disparities_used": disparities_before,
                "group_disparities_after": None if self.group_disparities_ is None else dict(self.group_disparities_),
            }

            block_summaries.append(block_summary)
            self.block_history_.append(block_summary)

        if not return_details:
            return final_preds_all

        return {
            "y_pred_final": final_preds_all,
            "y_pred_preliminary": preliminary_preds_all,
            "gad_preliminary_blocks": gad_preliminary_blocks,
            "gad_fair_blocks": gad_fair_blocks,
            "block_summaries": block_summaries,
            "final_group_disparities": None if self.group_disparities_ is None else dict(self.group_disparities_),
            "fairness_reports": list(self.fairness_reports_),
        }

    def reset_fairness_state(self) -> "LamdaFairGADClassifier":
        """
        Reinicia el estado fairness del modelo sin alterar el ajuste base LAMDA.

        Retorna
        -------
        LamdaFairGADClassifier
            La propia instancia con el estado fairness reiniciado.
        """
        self.group_disparities_ = None
        self.directional_disparities_ = None
        self.fairness_is_fitted_ = False

        self.y_true_history_ = None
        self.y_pred_history_ = None
        self.sensitive_history_ = None

        self.fairness_reports_ = []
        self.block_history_ = []

        return self

    def _predict_single_block(
        self,
        X_block: np.ndarray,
        sensitive_block: np.ndarray,
    ) -> dict[str, Any]:
        """
        Predice un único bloque usando las disparidades disponibles antes del bloque.

        Parámetros
        ----------
        X_block : np.ndarray
            Matriz de entrada del bloque.
        sensitive_block : np.ndarray
            Vector de grupos sensibles del bloque.

        Retorna
        -------
        dict[str, Any]
            Diccionario con GAD preliminar, GAD regularizado,
            predicciones preliminares, predicciones finales y
            disparidades utilizadas.
        """
        gad_preliminary = self.predict_gad(X_block)

        disparities_used = {}
        if self.group_disparities_ is not None:
            disparities_used = dict(self.group_disparities_)

        if self.disparity_mode == "directional":
            gad_fair = regularize_gad_directional(
                gad=gad_preliminary,
                sensitive=sensitive_block,
                classes=self.classes_,
                disparities=self.directional_disparities_ or {},
                lambda_=self.lambda_,
                clip=self.clip,
                default_group_disparity=self.default_group_disparity,
                only_preliminary_class=True,
            )
            preliminary_preds = np.argmax(gad_preliminary, axis=1)
            final_preds = np.argmax(gad_fair, axis=1)
        elif self.group_disparities_ is None:
            gad_fair, preliminary_preds, final_preds = regularize_gad_from_disparities(
                gad=gad_preliminary,
                sensitive=sensitive_block,
                group_disparities={},
                lambda_=self.lambda_,
                clip=self.clip,
                default_group_disparity=self.default_group_disparity,
                return_predictions=True,
            )
        else:
            gad_fair, preliminary_preds, final_preds = regularize_gad_from_disparities(
                gad=gad_preliminary,
                sensitive=sensitive_block,
                group_disparities=self.group_disparities_,
                lambda_=self.lambda_,
                clip=self.clip,
                default_group_disparity=self.default_group_disparity,
                return_predictions=True,
            )

        preliminary_labels = self.classes_[preliminary_preds]
        final_labels = self.classes_[final_preds]

        return {
            "gad_preliminary": gad_preliminary,
            "gad_fair": gad_fair,
            "preliminary_preds": preliminary_labels,
            "final_preds": final_labels,
            "group_disparities_used": disparities_used,
        }

    def _append_to_history(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Añade un bloque al histórico acumulado.

        Parámetros
        ----------
        y_true : np.ndarray
            Etiquetas reales del bloque.
        y_pred : np.ndarray
            Etiquetas predichas del bloque.
        sensitive : np.ndarray
            Grupos sensibles del bloque.

        Retorna
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Histórico acumulado actualizado.
        """
        if self.y_true_history_ is None:
            self.y_true_history_ = y_true.copy()
            self.y_pred_history_ = y_pred.copy()
            self.sensitive_history_ = sensitive.copy()
        else:
            self.y_true_history_ = np.concatenate([self.y_true_history_, y_true])
            self.y_pred_history_ = np.concatenate([self.y_pred_history_, y_pred])
            self.sensitive_history_ = np.concatenate([self.sensitive_history_, sensitive])

        return self.y_true_history_, self.y_pred_history_, self.sensitive_history_

    def _store_last_block_as_history(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> None:
        """
        Sustituye el histórico por el último bloque procesado.

        Parámetros
        ----------
        y_true : np.ndarray
            Etiquetas reales del bloque.
        y_pred : np.ndarray
            Etiquetas predichas del bloque.
        sensitive : np.ndarray
            Grupos sensibles del bloque.
        """
        self.y_true_history_ = y_true.copy()
        self.y_pred_history_ = y_pred.copy()
        self.sensitive_history_ = sensitive.copy()

    def _block_ranges(
        self,
        n_samples: int,
        block_size: int,
    ) -> list[tuple[int, int]]:
        """
        Genera los intervalos [start, end) asociados a cada bloque.

        Parámetros
        ----------
        n_samples : int
            Número total de individuos.
        block_size : int
            Tamaño de bloque.

        Retorna
        -------
        list[tuple[int, int]]
            Lista de pares (start, end) para cada bloque.
        """
        ranges: list[tuple[int, int]] = []

        for start in range(0, n_samples, block_size):
            end = min(start + block_size, n_samples)
            ranges.append((start, end))

        return ranges

    def _check_fairness_is_fitted(self) -> None:
        """
        Verifica que el modelo disponga de disparidades fairness calculadas.

        Lanza
        ------
        ValueError
            Si el estado fairness aún no ha sido ajustado.
        """
        if (
            not hasattr(self, "fairness_is_fitted_")
            or not self.fairness_is_fitted_
            or self.group_disparities_ is None
        ):
            raise ValueError(
                "El modelo no dispone todavía de disparidades fairness "
                "precalculadas. Ejecuta `update_fairness(...)` o "
                "`predict_blocks(...)` antes de usar este método."
            )