"""
Modelo LamdaFairMADClassifier.

Este módulo implementa una extensión de LAMDA clásico que incorpora
regularización fairness sobre MAD en un esquema iterativo por bloques.

Formulación
-----------
Sea MAD_{c,j}(X_r) el grado de adecuación marginal del descriptor j del
individuo r respecto a la clase c.

1. Impacto del descriptor
   Para cada descriptor j y grupo g se define:

       C_{j,g}^{(t)} = D_g^{(t)} - D_{g,(-j)}^{(t)}

   donde:
   - D_g^{(t)} es la disparidad del grupo g con el modelo completo
   - D_{g,(-j)}^{(t)} es la disparidad del grupo g cuando el descriptor j
     no participa en la agregación

2. Disparidad del descriptor
   Se considera el peor caso entre grupos:

       C_j^{(t)} = max_g C_{j,g}^{(t)}

   y se define:

       D_j^{(t)} = max(0, C_j^{(t)})

3. Peso fairness
   Para cada descriptor j se define:

       w_j^{(t)} = 1 / (1 + eta * max(0, D_j^{(t)} - tau_j))

4. Regularización de MAD
   El grado de adecuación marginal corregido se define como:

       MAD_{c,j}^{fair,(t)}(X_r) = (MAD_{c,j}(X_r))^{w_j^{(t-1)}}

5. Decisión final
   A partir del tensor MAD regularizado se obtiene GAD y se aplica la
   regla de decisión habitual de LAMDA.

Flujo
-----
1. Se ajusta el modelo base LAMDA.
2. Se divide el conjunto en bloques.
3. Para cada bloque t:
   - se calcula MAD preliminar,
   - se aplican los pesos fairness disponibles hasta t-1,
   - se obtiene la predicción final del bloque,
   - se actualiza el histórico,
   - se recalculan los pesos por descriptor cuando corresponde.
4. El primer bloque se predice sin regularización si aún no existen pesos.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lamda.core.decision import predict_class_labels
from lamda.fairness.regularization_mad_directional import regularize_mad_directional
from lamda.fairness.disparity import (
    FairnessThresholds,
    FairnessWeights,
    group_disparity_report,
)
from lamda.fairness.regularization_mad import (
    aggregate_mad_to_gad,
    apply_mad_fairness,
    compute_descriptor_weights,
    regularize_mad,
)
from lamda.models.lamda_classifier import LamdaClassifier


class LamdaFairMADClassifier(LamdaClassifier):
    """
    Clasificador LAMDA con regularización fairness sobre MAD.

    Parámetros
    ----------
    alpha : float, default=0.5
        Parámetro de exigencia del modelo base.
    operator : str, default="product"
        Operador de agregación GAD del modelo base.
    eta : float, default=1.0
        Intensidad de la penalización sobre descriptores.
    tau_j : float, default=0.0
        Umbral de disparidad por descriptor.
    block_size : int, default=128
        Tamaño de bloque usado en la predicción iterativa.
    update_every_n_blocks : int, default=1
        Frecuencia de actualización de los pesos fairness. Si vale 1, se
        actualizan tras cada bloque.
    fairness_thresholds : FairnessThresholds | None, default=None
        Umbrales de equidad.
    fairness_weights : FairnessWeights | None, default=None
        Pesos de agregación de fairness.
    clip : bool, default=True
        Si True, recorta los valores del tensor MAD regularizado.
    min_group_size : int, default=1
        Tamaño mínimo requerido para evaluar un grupo sensible.
    min_positive_size : int, default=1
        Número mínimo de ejemplos por clase dentro del grupo para evaluar EOD.
    default_group_disparity : float, default=0.0
        Disparidad asignada a grupos no evaluables.
    class_aggregation : str, default="mean"
        Estrategia de agregación sobre clases para calcular la disparidad.
    use_cumulative_history : bool, default=True
        Si True, las disparidades y pesos se calculan con histórico acumulado.

    Atributos
    ---------
    descriptor_weights_ : np.ndarray | None
        Vector de pesos fairness por descriptor.
    descriptor_disparities_ : np.ndarray | None
        Vector de disparidades efectivas por descriptor.
    descriptor_group_impacts_ : list[dict[Any, float]] | None
        Lista de impactos por grupo para cada descriptor.
    group_disparities_ : dict[Any, float] | None
        Disparidad vigente por grupo.
    fairness_is_fitted_ : bool
        Indica si el modelo dispone de pesos fairness calculados.
    X_history_ : np.ndarray | None
        Histórico acumulado de entradas.
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
        eta: float = 1.0,
        tau_j: float = 0.0,
        block_size: int = 128,
        update_every_n_blocks: int = 1,
        fairness_thresholds: FairnessThresholds | None = None,
        fairness_weights: FairnessWeights | None = None,
        clip: bool = True,
        min_group_size: int = 1,
        min_positive_size: int = 1,
        default_group_disparity: float = 0.0,
        class_aggregation: str = "mean",
        use_cumulative_history: bool = True,
        disparity_mode: str = "symmetric",
    ) -> None:
        """
        Inicializa el clasificador fairness sobre MAD.

        Parámetros
        ----------
        alpha : float, default=0.5
            Parámetro de exigencia del modelo base.
        operator : str, default="product"
            Operador de agregación GAD del modelo base.
        eta : float, default=1.0
            Intensidad de la penalización sobre descriptores.
        tau_j : float, default=0.0
            Umbral de disparidad por descriptor.
        block_size : int, default=128
            Tamaño de bloque usado en la predicción iterativa.
        update_every_n_blocks : int, default=1
            Frecuencia de actualización de los pesos fairness.
        fairness_thresholds : FairnessThresholds | None, default=None
            Umbrales de equidad.
        fairness_weights : FairnessWeights | None, default=None
            Pesos de agregación de fairness.
        clip : bool, default=True
            Si True, recorta los valores del tensor MAD regularizado.
        min_group_size : int, default=1
            Tamaño mínimo requerido para evaluar un grupo sensible.
        min_positive_size : int, default=1
            Número mínimo de ejemplos por clase dentro del grupo para evaluar EOD.
        default_group_disparity : float, default=0.0
            Disparidad asignada a grupos no evaluables.
        class_aggregation : str, default="mean"
            Estrategia de agregación sobre clases.
        use_cumulative_history : bool, default=True
            Si True, se usa histórico acumulado.
        """
        super().__init__(alpha=alpha, operator=operator)

        self.eta = eta
        self.tau_j = tau_j
        self.block_size = block_size
        self.update_every_n_blocks = update_every_n_blocks
        self.fairness_thresholds = fairness_thresholds or FairnessThresholds()
        self.fairness_weights = fairness_weights or FairnessWeights()
        self.clip = clip
        self.min_group_size = min_group_size
        self.min_positive_size = min_positive_size
        self.default_group_disparity = default_group_disparity
        self.class_aggregation = class_aggregation
        self.use_cumulative_history = use_cumulative_history
        # Señal de control empleada al medir el impacto de cada descriptor.
        # Por defecto simétrica: el peso resultante es único y global por
        # descriptor, de modo que la dirección de la disparidad no tiene
        # destinatario al que aplicarse. El modo direccional se conserva para
        # el análisis de sensibilidad.
        if disparity_mode not in ("directional", "symmetric"):
            raise ValueError("disparity_mode debe ser 'directional' o 'symmetric'.")
        self.disparity_mode = disparity_mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LamdaFairMADClassifier":
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
        LamdaFairMADClassifier
            La propia instancia ajustada.
        """
        super().fit(X, y)

        self.descriptor_weights_ = None
        self.descriptor_disparities_ = None
        self.descriptor_group_impacts_ = None
        self.group_disparities_ = None
        self.fairness_is_fitted_ = False

        self.X_history_ = None
        self.y_true_history_ = None
        self.y_pred_history_ = None
        self.sensitive_history_ = None

        self.fairness_reports_ = []
        self.block_history_ = []

        return self

    def update_fairness(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        sensitive: np.ndarray,
    ) -> "LamdaFairMADClassifier":
        """
        Actualiza los pesos fairness por descriptor a partir de datos observados.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada observada.
        y_true : np.ndarray
            Etiquetas reales observadas.
        sensitive : np.ndarray
            Vector de grupos sensibles.

        Retorna
        -------
        LamdaFairMADClassifier
            La propia instancia con el estado fairness actualizado.
        """
        self._check_is_fitted()

        X = np.asarray(X)
        y_true = np.asarray(y_true).reshape(-1)
        sensitive = np.asarray(sensitive).reshape(-1)

        if self.use_cumulative_history:
            X_ref, y_true_ref, y_pred_ref, sensitive_ref = self._append_to_history(
                X=X,
                y_true=y_true,
                sensitive=sensitive,
            )
        else:
            y_pred_current = self.predict(X)
            self._store_last_block_as_history(
                X=X,
                y_true=y_true,
                y_pred=y_pred_current,
                sensitive=sensitive,
            )
            X_ref = self.X_history_
            y_true_ref = self.y_true_history_
            y_pred_ref = self.y_pred_history_
            sensitive_ref = self.sensitive_history_

        return self._fit_fairness_from(
            X_ref=X_ref,
            y_true_ref=y_true_ref,
            sensitive_ref=sensitive_ref,
        )

    def _fit_fairness_from(
        self,
        X_ref: np.ndarray,
        y_true_ref: np.ndarray,
        sensitive_ref: np.ndarray,
    ) -> "LamdaFairMADClassifier":
        """
        Calcula el estado fairness (pesos y disparidades por descriptor) a partir
        de un conjunto de referencia ya acumulado, sin volver a incorporarlo al
        histórico.

        Se separa de `update_fairness` para poder recalcular el estado al final
        del recorrido por bloques sin duplicar observaciones en el histórico.
        """
        mad_ref = self.predict_mad(X_ref)

        regularizador = (regularize_mad_directional
                         if self.disparity_mode == "directional"
                         else regularize_mad)

        result = regularizador(
            mad=mad_ref,
            y_true=y_true_ref,
            sensitive=sensitive_ref,
            classes=self.classes_,
            alpha=self.alpha,           
            operator=self.operator,
            thresholds=self.fairness_thresholds,
            weights=self.fairness_weights,
            eta=self.eta,
            tau_j=self.tau_j,
            min_group_size=self.min_group_size,
            min_positive_size=self.min_positive_size,
            default_for_small_groups=self.default_group_disparity,
            class_aggregation=self.class_aggregation,
            clip=self.clip,
            return_details=True,
)

        self.descriptor_weights_ = result["descriptor_weights"]
        self.descriptor_disparities_ = result["descriptor_disparities"]
        self.descriptor_group_impacts_ = result["descriptor_group_impacts"]
        self.group_disparities_ = result["baseline_group_disparities"]

        report = group_disparity_report(
            y_true=y_true_ref,
            y_pred=result["y_pred_base"],
            sensitive=sensitive_ref,
            thresholds=self.fairness_thresholds,
            weights=self.fairness_weights,
            class_aggregation=self.class_aggregation,
        )

        self.fairness_reports_.append(report)
        self.fairness_is_fitted_ = True

        return self

    def predict_mad_fair_from_weights(
        self,
        X: np.ndarray,
        clip: bool | None = None,
    ) -> np.ndarray:
        """
        Calcula el tensor MAD regularizado a partir de los pesos almacenados.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada.
        clip : bool | None, default=None
            Si True, recorta los valores del tensor MAD regularizado.

        Retorna
        -------
        np.ndarray
            Tensor MAD regularizado.
        """
        self._check_is_fitted()
        self._check_fairness_is_fitted()

        clip = self.clip if clip is None else clip

        mad = self.predict_mad(X)

        return apply_mad_fairness(
            mad=mad,
            descriptor_weights=self.descriptor_weights_,
            clip=clip,
        )

    def predict_fair_from_weights(
            self,
            X: np.ndarray,
            clip: bool | None = None,
        ) -> np.ndarray:
            """
            Predice etiquetas usando los pesos fairness almacenados en el modelo.

            Parámetros
            ----------
            X : np.ndarray
                Matriz de entrada.
            clip : bool | None, default=None
                Si True, recorta los valores del tensor MAD regularizado.

            Retorna
            -------
            np.ndarray
                Vector con etiquetas predichas.
            """
            mad_fair = self.predict_mad_fair_from_weights(X=X, clip=clip)
            gad_fair = aggregate_mad_to_gad(mad=mad_fair, alpha=self.alpha, operator=self.operator)
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
        gad_base_blocks: list[np.ndarray] = []
        gad_fair_blocks: list[np.ndarray] = []
        mad_fair_blocks: list[np.ndarray] = []
        block_summaries: list[dict[str, Any]] = []

        for block_index, (start, end) in enumerate(self._block_ranges(n_samples, block_size)):
            X_block = X[start:end]
            y_block = y_true[start:end]
            s_block = sensitive[start:end]

            block_result = self._predict_single_block(X_block=X_block)

            final_block = block_result["final_preds"]
            final_preds_all[start:end] = final_block

            gad_base_blocks.append(block_result["gad_base"])
            gad_fair_blocks.append(block_result["gad_fair"])
            mad_fair_blocks.append(block_result["mad_fair"])

            if (block_index + 1) % self.update_every_n_blocks == 0:
                self.update_fairness(
                    X=X_block,
                    y_true=y_block,
                    sensitive=s_block,
                )
            else:
                self._append_to_history(
                    X=X_block,
                    y_true=y_block,
                    sensitive=s_block,
                    y_pred_override=final_block,
                )

            block_summary = {
                "block_index": block_index,
                "start": start,
                "end": end,
                "n_samples": end - start,
                "weights_used": None if self.descriptor_weights_ is None else self.descriptor_weights_.copy(),
                "group_disparities_after": None if self.group_disparities_ is None else dict(self.group_disparities_),
            }

            block_summaries.append(block_summary)
            self.block_history_.append(block_summary)

        # Actualización final garantizada. Si el calendario de actualización no
        # llegó a dispararse en ningún bloque (situación que se produce cuando
        # update_every_n_blocks supera el número de bloques disponibles), el
        # modelo terminaría el recorrido sin estado fairness y la predicción
        # posterior fallaría. Se recalcula entonces el estado sobre el histórico
        # acumulado, sin incorporar observaciones nuevas. Esto solo define el
        # comportamiento en la frontera del calendario; no altera el mecanismo
        # de regularización ni el orden temporal de la información utilizada.
        if self.descriptor_weights_ is None and self.X_history_ is not None:
            self._fit_fairness_from(
                X_ref=self.X_history_,
                y_true_ref=self.y_true_history_,
                sensitive_ref=self.sensitive_history_,
            )

        if not return_details:
            return final_preds_all

        return {
            "y_pred_final": final_preds_all,
            "gad_base_blocks": gad_base_blocks,
            "gad_fair_blocks": gad_fair_blocks,
            "mad_fair_blocks": mad_fair_blocks,
            "block_summaries": block_summaries,
            "descriptor_weights": None if self.descriptor_weights_ is None else self.descriptor_weights_.copy(),
            "descriptor_disparities": None if self.descriptor_disparities_ is None else self.descriptor_disparities_.copy(),
            "final_group_disparities": None if self.group_disparities_ is None else dict(self.group_disparities_),
            "fairness_reports": list(self.fairness_reports_),
        }

    def reset_fairness_state(self) -> "LamdaFairMADClassifier":
        """
        Reinicia el estado fairness del modelo sin alterar el ajuste base LAMDA.

        Retorna
        -------
        LamdaFairMADClassifier
            La propia instancia con el estado fairness reiniciado.
        """
        self.descriptor_weights_ = None
        self.descriptor_disparities_ = None
        self.descriptor_group_impacts_ = None
        self.group_disparities_ = None
        self.fairness_is_fitted_ = False

        self.X_history_ = None
        self.y_true_history_ = None
        self.y_pred_history_ = None
        self.sensitive_history_ = None

        self.fairness_reports_ = []
        self.block_history_ = []

        return self

    def _predict_single_block(
        self,
        X_block: np.ndarray,
    ) -> dict[str, Any]:
        """
        Predice un único bloque usando los pesos disponibles antes del bloque.

        Parámetros
        ----------
        X_block : np.ndarray
            Matriz de entrada del bloque.

        Retorna
        -------
        dict[str, Any]
            Diccionario con MAD base, MAD regularizado, GAD base,
            GAD regularizado y predicciones finales.
        """
        mad_base = self.predict_mad(X_block)
        gad_base = aggregate_mad_to_gad(mad=mad_base, alpha=self.alpha, operator=self.operator)  # <- alpha añadido

        if self.descriptor_weights_ is None:
            mad_fair = mad_base.copy()
        else:
            mad_fair = apply_mad_fairness(
                mad=mad_base,
                descriptor_weights=self.descriptor_weights_,
                clip=self.clip,
            )

        gad_fair = aggregate_mad_to_gad(mad=mad_fair, alpha=self.alpha, operator=self.operator)  # <- alpha añadido
        final_preds = predict_class_labels(gad_fair, self.classes_)

        return {
            "mad_base": mad_base,
            "mad_fair": mad_fair,
            "gad_base": gad_base,
            "gad_fair": gad_fair,
            "final_preds": final_preds,
        }

    def _append_to_history(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        sensitive: np.ndarray,
        y_pred_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Añade un bloque al histórico acumulado.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada del bloque.
        y_true : np.ndarray
            Etiquetas reales del bloque.
        sensitive : np.ndarray
            Grupos sensibles del bloque.
        y_pred_override : np.ndarray | None, default=None
            Predicciones del bloque. Si es None, se obtienen con el modelo
            fairness actual si está disponible; en caso contrario, con el modelo base.

        Retorna
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Histórico acumulado actualizado.
        """
        if y_pred_override is None:
            if self.descriptor_weights_ is None:
                y_pred = self.predict(X)
            else:
                y_pred = self.predict_fair_from_weights(X)
        else:
            y_pred = np.asarray(y_pred_override).reshape(-1)

        if self.X_history_ is None:
            self.X_history_ = X.copy()
            self.y_true_history_ = y_true.copy()
            self.y_pred_history_ = y_pred.copy()
            self.sensitive_history_ = sensitive.copy()
        else:
            self.X_history_ = np.concatenate([self.X_history_, X], axis=0)
            self.y_true_history_ = np.concatenate([self.y_true_history_, y_true])
            self.y_pred_history_ = np.concatenate([self.y_pred_history_, y_pred])
            self.sensitive_history_ = np.concatenate([self.sensitive_history_, sensitive])

        return (
            self.X_history_,
            self.y_true_history_,
            self.y_pred_history_,
            self.sensitive_history_,
        )

    def _store_last_block_as_history(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> None:
        """
        Sustituye el histórico por el último bloque procesado.

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada del bloque.
        y_true : np.ndarray
            Etiquetas reales del bloque.
        y_pred : np.ndarray
            Etiquetas predichas del bloque.
        sensitive : np.ndarray
            Grupos sensibles del bloque.
        """
        self.X_history_ = X.copy()
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
        Verifica que el modelo disponga de pesos fairness calculados.

        Lanza
        ------
        ValueError
            Si el estado fairness aún no ha sido ajustado.
        """
        if (
            not hasattr(self, "fairness_is_fitted_")
            or not self.fairness_is_fitted_
            or self.descriptor_weights_ is None
        ):
            raise ValueError(
                "El modelo no dispone todavía de pesos fairness "
                "precalculados. Ejecuta `update_fairness(...)` o "
                "`predict_blocks(...)` antes de usar este método."
            )