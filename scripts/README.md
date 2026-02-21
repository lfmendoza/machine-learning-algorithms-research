# Tarea 2 — Otros Algoritmos de Aprendizaje No Supervisado

Scripts para el laboratorio de Minería de Datos: SVD, t-SNE, UMAP e ICA
aplicados a conjuntos de datos reales.

## Estructura del proyecto

```
scripts/
  requirements.txt         Dependencias Python
  download_physionet.py    Descarga datos ECG desde PhysioNet
  svd_movielens.py         SVD sobre MovieLens 100k
  tsne_umap_cancer.py      t-SNE y UMAP sobre Breast Cancer Wisconsin
  ica_ecg.py               ICA sobre señales ECG (MIT-BIH P-Wave)
  run_all.py               Ejecuta todos los scripts en secuencia
  run_avance.py            (Script del avance anterior — no se usa)
datasets/
  movie-lens/              MovieLens 100k (u.data, u.item, ...)
  brest-cancer/            Breast Cancer Wisconsin (data.csv)
  physionet-ecg/           Señales ECG descargadas (creado por download_physionet.py)
outputs/
  svd/                     Figuras y tablas de SVD
  tsne/                    Figuras y tablas de t-SNE
  umap/                    Figuras y tablas de UMAP
  ica/                     Figuras y tablas de ICA
```

## Instalación

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r scripts/requirements.txt
```

## Ejecución rápida (todos los algoritmos)

```bash
python scripts/run_all.py --seed 42
```

Esto ejecuta en orden: descarga de datos PhysioNet, SVD, t-SNE + UMAP e ICA.

## Ejecución individual

### 1. Descargar datos ECG (PhysioNet)

```bash
python scripts/download_physionet.py
```

Descarga los 12 registros del MIT-BIH Arrhythmia Database P-Wave Annotations
(~23 MB) en `datasets/physionet-ecg/`. Solo descarga archivos faltantes.

### 2. SVD — MovieLens 100k

```bash
python scripts/svd_movielens.py --components 50 --seed 42
```

**Dataset**: MovieLens 100k (943 usuarios, 1682 películas, 100k ratings).

**Salidas** (`outputs/svd/`):
- `fig_svd_01_varianza_explicada.png` — Varianza explicada por componente y acumulada
- `fig_svd_02_usuarios_2d.png` — Proyección de usuarios en 2 componentes
- `fig_svd_03_peliculas_2d.png` — Proyección de películas coloreada por género
- `fig_svd_04_reconstruccion_error.png` — Error de reconstrucción vs. componentes
- `svd_varianza_explicada.csv` — Tabla de varianza explicada
- `svd_top_peliculas_por_componente.csv` — Top 10 películas por factor latente
- `resumen_svd.md` — Resumen para el informe

### 3. t-SNE y UMAP — Breast Cancer Wisconsin

```bash
python scripts/tsne_umap_cancer.py --seed 42
```

**Dataset**: Breast Cancer Wisconsin (569 muestras, 30 features, diagnóstico M/B).

**Salidas t-SNE** (`outputs/tsne/`):
- `fig_tsne_01_perplexity_comparison.png` — Grid de 4 perplexidades
- `fig_tsne_02_best_projection.png` — Mejor proyección individual
- `tsne_params_silhouette.csv` — Tabla de sensibilidad de parámetros

**Salidas UMAP** (`outputs/umap/`):
- `fig_umap_01_neighbors_comparison.png` — Grid de n_neighbors
- `fig_umap_02_mindist_comparison.png` — Efecto de min_dist
- `fig_umap_03_best_projection.png` — Mejor proyección individual
- `fig_comparison_tsne_vs_umap.png` — Comparación lado a lado
- `umap_params_silhouette.csv` — Tabla de sensibilidad

### 4. ICA — Señales ECG (MIT-BIH P-Wave)

```bash
python scripts/ica_ecg.py --seed 42
```

**Dataset**: MIT-BIH Arrhythmia Database P-Wave Annotations (12 registros ECG,
2 canales, 360 Hz).

**Salidas** (`outputs/ica/`):
- `fig_ica_01_originales_*.png` — Señales ECG originales
- `fig_ica_02_componentes_*.png` — Componentes independientes (FastICA)
- `fig_ica_03_comparacion_*.png` — Original vs ICA lado a lado
- `fig_ica_04_kurtosis.png` — Kurtosis comparativa (originales vs ICA)
- `fig_ica_05_pwave_*.png` — Detalle con anotaciones P-wave
- `ica_kurtosis.csv` — Tabla de kurtosis por registro y componente
- `ica_mixing_matrix_*.csv` — Matriz de mezcla estimada por registro

## Parámetros comunes

| Parámetro | Script | Default | Descripción |
|-----------|--------|---------|-------------|
| `--seed`  | todos  | 42      | Semilla para reproducibilidad |
| `--output`| todos  | `outputs/<algo>` | Directorio de salida |
| `--components` | svd_movielens | 50 | Número de componentes SVD |
| `--records` | ica_ecg | 100 119 207 | Registros ECG a procesar |
| `--window` | ica_ecg | 3600 | Muestras por ventana (10s a 360Hz) |

## Reproducibilidad

- Todos los scripts fijan `random_state` via `--seed` (default 42).
- Las versiones de librerías se fijan en `requirements.txt`.
- Para resultados idénticos, mantenga constantes: seed, versiones, datasets.

## Correspondencia con el informe

| Sección del informe | Script | Resumen generado |
|---------------------|--------|------------------|
| 2.1 SVD             | `svd_movielens.py` | `outputs/svd/resumen_svd.md` |
| 2.2 t-SNE           | `tsne_umap_cancer.py` | `outputs/tsne/resumen_tsne.md` |
| 2.3 UMAP            | `tsne_umap_cancer.py` | `outputs/umap/resumen_umap.md` |
| 2.4 ICA             | `ica_ecg.py` | `outputs/ica/resumen_ica.md` |
| 3 Comparación       | `tsne_umap_cancer.py` | `fig_comparison_tsne_vs_umap.png` |
