"""
Perfiles de métricas de equidad por tipo de dataset.

Este módulo centraliza la decisión de qué métricas de equidad intervienen en
el cálculo de la disparidad D_g y con qué estrategia de agregación, en función
de la naturaleza del problema (binario o multiclase).

Motivación
----------
La disparidad D_g que gobierna la penalización de Fair-GAD y Fair-MAD se
construye combinando SPD, EOD y DI simétrico bajo unos umbrales. Sin embargo:

- En clasificación binaria, las tres métricas (SPD, EOD, DI) son estándar y
  comparables con la literatura de referencia (por ejemplo, Le Quy et al.,
  2022, que reporta SPD y EOD sobre estos mismos datasets con regresión
  logística).
- En clasificación multiclase, DI (cociente de tasas de resultado favorable)
  deja de ser una métrica estándar, ya que el resultado favorable no es único.
  La literatura de equidad multiclase (por ejemplo, Alghamdi et al., 2022)
  emplea extensiones de SPD y de equalized odds (MEO), pero no DI. Por
  coherencia y comparabilidad, en multiclase se desactiva DI.

Diseño (Opción A)
-----------------
El perfil gobierna tanto la señal de penalización D_g como el reporte de
disparidad agregada, de modo que penalización y reporte comparten construcción
y son mutuamente coherentes. La agregación sobre clases se mantiene en "mean"
también en multiclase, por estabilidad de la penalización (especialmente de
Fair-GAD). El MEO por máximo, necesario para comparar con la literatura
multiclase, se calcula aparte en las funciones de reporte crudo del módulo de
métricas y NO gobierna la penalización.

Uso
---
    from lamda.fairness.metric_profiles import build_metric_profile

    # Selección automática por número de clases:
    profile = build_metric_profile(n_classes=2)          # perfil binario
    profile = build_metric_profile(n_classes=5)          # perfil multiclase

    # Override manual explícito:
    profile = build_metric_profile(kind="binary")
    profile = build_metric_profile(kind="multiclass")

    # El perfil se pasa a los clasificadores:
    clf = LamdaFairGADClassifier(
        fairness_weights=profile.weights,
        class_aggregation=profile.class_aggregation,
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lamda.fairness.disparity import FairnessWeights


# =========================================================
# DEFINICIÓN DEL PERFIL
# =========================================================

@dataclass(frozen=True)
class MetricProfile:
    """
    Perfil de métricas de equidad para un tipo de dataset.

    Atributos
    ---------
    kind : str
        Tipo de perfil ("binary" o "multiclass").
    weights : FairnessWeights
        Pesos de agregación de las componentes de D_g. En el perfil
        multiclase, el peso de DI es 0 (DI queda desactivado).
    class_aggregation : str
        Estrategia de agregación sobre clases usada para construir D_g.
        Se mantiene "mean" en ambos perfiles por estabilidad (Opción A).
    active_metrics : tuple[str, ...]
        Métricas activas en la penalización. Sirve como registro explícito
        y trazable de qué entra en D_g, útil para documentar el experimento.
    report_metrics : tuple[str, ...]
        Métricas que deben mostrarse en las tablas de detección para ser
        comparables con la literatura. No gobiernan D_g; son el escaparate
        comparable. En multiclase incluye MEO (equalized odds por máximo).
    """

    kind: str
    weights: FairnessWeights
    class_aggregation: str
    active_metrics: tuple[str, ...]
    report_metrics: tuple[str, ...]


# =========================================================
# CONSTRUCTORES DE PERFIL
# =========================================================

def binary_profile() -> MetricProfile:
    """
    Construye el perfil de métricas para clasificación binaria.

    En binario se emplean SPD, EOD y DI simétrico, las tres con peso 1,
    y agregación "mean" (irrelevante al haber una única clase favorable).
    Es el perfil comparable con la literatura binaria (SPD/EOD de Le Quy
    et al., 2022, y DI en trabajos que lo reportan).

    Retorna
    -------
    MetricProfile
        Perfil binario.
    """
    return MetricProfile(
        kind="binary",
        weights=FairnessWeights(spd=1.0, eod=1.0, di=1.0),
        class_aggregation="mean",
        #active_metrics=("spd", "eod", "di_sym"),
        active_metrics=("spd"),
        report_metrics=("spd", "eod", "di"),
    )


def multiclass_profile() -> MetricProfile:
    """
    Construye el perfil de métricas para clasificación multiclase.

    En multiclase se emplean SPD y EOD (peso 1) y se desactiva DI (peso 0),
    porque DI no es una métrica multiclase estándar. La agregación se mantiene
    en "mean" para la penalización (Opción A). El reporte comparable con la
    literatura multiclase (Alghamdi et al., 2022) incluye SPD multiclase y MEO
    (equalized odds por máximo), calculados aparte en el módulo de métricas.

    Retorna
    -------
    MetricProfile
        Perfil multiclase.
    """
    return MetricProfile(
        kind="multiclass",
        weights=FairnessWeights(spd=1.0, eod=1.0, di=0.0),
        class_aggregation="mean",
        #active_metrics=("spd", "eod"),
        active_metrics=("spd_multiclass"),
        report_metrics=("spd_multiclass", "meo"),
    )


def build_metric_profile(
    n_classes: int | None = None,
    kind: str | None = None,
    multiclass_threshold: int = 3,
) -> MetricProfile:
    """
    Construye el perfil de métricas de forma automática o manual.

    La selección es automática por número de clases cuando se pasa `n_classes`
    y no se fuerza `kind`. El override manual mediante `kind` tiene prioridad
    y permite, por ejemplo, correr un experimento de ablación forzando el perfil
    binario (con DI) sobre un dataset multiclase.

    Parámetros
    ----------
    n_classes : int | None, default=None
        Número de clases del problema. Necesario si no se especifica `kind`.
    kind : str | None, default=None
        Fuerza el perfil ("binary" o "multiclass"). Si se indica, tiene
        prioridad sobre `n_classes`.
    multiclass_threshold : int, default=3
        Número de clases a partir del cual (inclusive) el problema se considera
        multiclase. Con el valor por defecto, 2 clases -> binario y 3 o más ->
        multiclase.

    Retorna
    -------
    MetricProfile
        Perfil seleccionado.

    Lanza
    ------
    ValueError
        Si no se proporciona ni `n_classes` ni `kind`, o si `kind` no es válido.
    """
    if kind is not None:
        if kind == "binary":
            return binary_profile()
        if kind == "multiclass":
            return multiclass_profile()
        raise ValueError("kind debe ser 'binary' o 'multiclass'.")

    if n_classes is None:
        raise ValueError(
            "Debe indicarse 'n_classes' para la selección automática, "
            "o bien forzar 'kind'."
        )

    if n_classes < 2:
        raise ValueError("n_classes debe ser al menos 2.")

    if n_classes < multiclass_threshold:
        return binary_profile()

    return multiclass_profile()
