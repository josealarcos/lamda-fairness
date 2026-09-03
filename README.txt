# LAMDA — Detección y mitigación de sesgos en clasificación difusa

Implementación de LAMDA clásico y dos variantes de equidad:
**Fair-GAD** (regularización sobre el grado de adecuación global) y
**Fair-MAD** (regularización sobre los grados de adecuación marginal).

## Estructura
- `src/lamda/` — paquete: núcleo (MAD, GAD, decisión, normalización, prototipos),
  métricas de equidad, regularización GAD/MAD, clasificadores y carga de datos.
- `notebooks/` — experimentos: detección, mitigación, baselines, exploración de
  parámetros, extensión interseccional y discusión agregada.
- `notebooks/resultados/` — salidas que consumen los notebooks agregados.
- `data/raw/` — datasets (adult, communities, compas, dutch, oulad, student).

## Requisitos
    pip install -r requirements.txt
