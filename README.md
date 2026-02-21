# Tarea 2 — Otros Algoritmos de Aprendizaje No Supervisado

**Asignatura**: Minería de Datos
**Fecha**: Febrero 2026

---

## 1. Introducción

### Contexto general del aprendizaje no supervisado

El aprendizaje no supervisado comprende un conjunto de técnicas de análisis de datos que buscan descubrir patrones, estructuras o relaciones en datos sin etiquetas predefinidas. A diferencia del aprendizaje supervisado, no se dispone de una variable objetivo; el algoritmo debe identificar por sí mismo la organización inherente de los datos.

Entre las tareas principales del aprendizaje no supervisado se encuentran:

- **Reducción de dimensionalidad**: representar datos de alta dimensión en espacios de menor dimensión preservando la información más relevante (SVD, PCA, ICA).
- **Visualización**: proyectar datos multidimensionales a 2D o 3D para exploración visual (t-SNE, UMAP).
- **Separación de fuentes**: descomponer señales observadas en componentes independientes subyacentes (ICA).

Estas técnicas son fundamentales en el análisis moderno de datos, ya que permiten explorar, comprender y preprocesar conjuntos de datos complejos antes de aplicar modelos predictivos o de toma de decisiones.

### Objetivos del trabajo

1. Comprender el funcionamiento teórico de cuatro algoritmos de aprendizaje no supervisado: SVD, t-SNE, UMAP e ICA.
2. Identificar los principales usos, aplicaciones y limitaciones de cada algoritmo.
3. Aplicar cada algoritmo a un conjunto de datos real, analizando e interpretando los resultados obtenidos.
4. Comparar los algoritmos entre sí, destacando ventajas, limitaciones y contextos de uso apropiados.

---

## 2. Desarrollo

Cada algoritmo se aplica a un dataset específico según los requerimientos del laboratorio. Los reportes detallados (descripción teórica, usos, resultados e interpretación) se generan automáticamente al ejecutar los scripts y se encuentran en la carpeta `outputs/`.

### 2.1 SVD (Descomposición en Valores Singulares)

- **Descripción teórica**: factorización matricial A = U·Σ·Vᵀ para reducción de dimensionalidad.
- **Usos y aplicaciones**: sistemas de recomendación, compresión de imágenes, procesamiento de lenguaje natural.
- **Dataset utilizado**: MovieLens 100k (943 usuarios, 1682 películas, 100,000 ratings).
- **Resultados**: varianza explicada por componente, proyecciones 2D de usuarios y películas, error de reconstrucción.
- **Interpretación**: ver `outputs/svd/resumen_svd.md`.

### 2.2 t-SNE (Incrustación Estocástica de Vecinos con Distribución t)

- **Descripción teórica**: reducción no lineal que preserva la estructura local mediante distribuciones de probabilidad y minimización de la divergencia KL.
- **Usos y aplicaciones**: visualización exploratoria, validación de agrupamientos, análisis de representaciones de redes neuronales.
- **Dataset utilizado**: Breast Cancer Wisconsin (569 muestras, 30 características, diagnóstico M/B).
- **Resultados**: proyecciones 2D con distintas perplejidades, métricas de silueta y divergencia KL.
- **Interpretación**: ver `outputs/tsne/resumen_tsne.md`.

### 2.3 UMAP (Aproximación y Proyección Uniforme de Variedades)

- **Descripción teórica**: reducción no lineal basada en topología algebraica que preserva estructura local y global.
- **Usos y aplicaciones**: visualización de alta dimensión, preprocesamiento para agrupamiento, exploración de representaciones vectoriales.
- **Dataset utilizado**: Breast Cancer Wisconsin (mismo dataset que t-SNE para comparación directa).
- **Resultados**: proyecciones 2D variando n_neighbors y min_dist, comparación lado a lado con t-SNE.
- **Interpretación**: ver `outputs/umap/resumen_umap.md`.

### 2.4 ICA (Análisis de Componentes Independientes)

- **Descripción teórica**: separación ciega de fuentes que recupera señales independientes a partir de mezclas lineales, maximizando la no-gaussianidad.
- **Usos y aplicaciones**: procesamiento de señales biomédicas (ECG, EEG), eliminación de artefactos, problema del cóctel.
- **Dataset utilizado**: MIT-BIH Arrhythmia Database P-Wave Annotations (12 registros ECG, 2 canales, 360 Hz).
- **Resultados**: componentes independientes separados, análisis de kurtosis, detalle con anotaciones de onda P.
- **Interpretación**: ver `outputs/ica/resumen_ica.md`.

---

## 3. Comparación general

### Comparación entre algoritmos

| Aspecto | SVD | t-SNE | UMAP | ICA |
|---|---|---|---|---|
| Tipo | Lineal | No lineal | No lineal | Lineal |
| Objetivo | Reducir dimensionalidad (varianza) | Visualización 2D/3D | Visualización + reducción | Separar fuentes independientes |
| Preserva | Varianza global | Estructura local | Local + global | Independencia estadística |
| Escalabilidad | Excelente | Limitada (O(n²)) | Buena (O(n^1.14)) | Buena |
| Datos nuevos | Sí (transformación directa) | No | Sí (.transform()) | Sí (matriz W estimada) |
| Supuesto clave | Ninguno especial | No paramétrico | Datos sobre un manifold | Fuentes no-gaussianas e independientes |
| Determinismo | Pseudoaleatorio (solver) | Estocástico | Estocástico | Pseudoaleatorio (inicialización) |

### Ventajas y limitaciones

| Algoritmo | Ventajas | Limitaciones |
|---|---|---|
| SVD | Eficiente con matrices dispersas; interpretable (factores latentes); base matemática sólida | No captura relaciones no lineales; sensible a valores extremos |
| t-SNE | Excelente para visualización de agrupamientos; revela estructura local fina | Lento para datasets grandes; no preserva distancias globales; no permite transformar datos nuevos |
| UMAP | Rápido; preserva estructura global y local; soporta datos nuevos | Resultados dependen de hiperparámetros; fundamento teórico más complejo |
| ICA | Separa fuentes independientes; útil para señales biomédicas; interpretable físicamente | Requiere al menos tantos sensores como fuentes; asume mezcla lineal e instantánea |

---

## 4. Conclusiones

### Principales aprendizajes

- Cada algoritmo tiene un propósito distinto dentro del aprendizaje no supervisado: SVD para factorización y reducción, t-SNE y UMAP para visualización, e ICA para separación de fuentes.
- La elección del algoritmo depende del objetivo del análisis: si se busca comprimir información (SVD), explorar visualmente (t-SNE/UMAP) o descomponer señales mixtas (ICA).
- Los hiperparámetros (perplejidad en t-SNE, n_neighbors en UMAP, número de componentes en SVD) tienen un impacto significativo en los resultados y deben explorarse sistemáticamente.

### Dificultades encontradas

- La descarga y lectura de datos en formato WFDB (PhysioNet) requiere familiarizarse con la librería `wfdb` y el formato de anotaciones.
- t-SNE es computacionalmente costoso y sensible a la perplejidad; requiere experimentar con varios valores para obtener visualizaciones informativas.
- La interpretación de los componentes ICA en señales ECG requiere conocimiento del dominio (cardiología) para validar si la separación de fuentes tiene sentido fisiológico.

### Reflexión final del grupo

Los algoritmos de aprendizaje no supervisado son herramientas complementarias, no competidoras. En un flujo de trabajo real de análisis de datos, es común usar SVD para preprocesar y comprimir, t-SNE o UMAP para explorar visualmente los patrones descubiertos, e ICA cuando se necesita separar señales mezcladas. La comprensión de sus fundamentos teóricos, supuestos y limitaciones es esencial para seleccionar la técnica adecuada y evitar interpretaciones erróneas de los resultados.

---

## 5. Referencias

1. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
2. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
3. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
4. Hyvärinen, A., & Oja, E. (2000). Independent Component Analysis: Algorithms and Applications. *Neural Networks*, 13(4–5), 411–430.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
6. GroupLens Research. MovieLens 100K Dataset. https://grouplens.org/datasets/movielens/100k/
7. UCI Machine Learning Repository. Breast Cancer Wisconsin (Diagnostic) Data Set. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
8. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220. https://physionet.org/
9. Llamedo, M., & Martínez, J.P. (2018). MIT-BIH Arrhythmia Database P-Wave Annotations. PhysioNet. https://physionet.org/content/pwave/1.0.0/
10. Documentación scikit-learn: https://scikit-learn.org/stable/
11. Documentación UMAP: https://umap-learn.readthedocs.io/
12. Documentación wfdb-python: https://wfdb.readthedocs.io/

---

## Estructura del proyecto

```
README.md                    Este archivo
scripts/
  requirements.txt           Dependencias Python
  download_physionet.py      Descarga datos ECG desde PhysioNet
  svd_movielens.py           SVD sobre MovieLens 100k
  tsne_umap_cancer.py        t-SNE y UMAP sobre Breast Cancer Wisconsin
  ica_ecg.py                 ICA sobre señales ECG (MIT-BIH P-Wave)
  run_all.py                 Ejecuta todos los scripts en secuencia
datasets/
  movie-lens/                MovieLens 100k (u.data, u.item)
  brest-cancer/              Breast Cancer Wisconsin (data.csv)
  physionet-ecg/             Señales ECG (creado por download_physionet.py)
outputs/
  svd/                       Figuras, tablas y resumen de SVD
  tsne/                      Figuras, tablas y resumen de t-SNE
  umap/                      Figuras, tablas y resumen de UMAP
  ica/                       Figuras, tablas y resumen de ICA
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

### Descargar datos ECG (PhysioNet)

```bash
python scripts/download_physionet.py
```

Descarga los 12 registros del MIT-BIH Arrhythmia Database P-Wave Annotations
(~23 MB) en `datasets/physionet-ecg/`. Solo descarga archivos faltantes.

### SVD — MovieLens 100k

```bash
python scripts/svd_movielens.py --components 50 --seed 42
```

### t-SNE y UMAP — Breast Cancer Wisconsin

```bash
python scripts/tsne_umap_cancer.py --seed 42
```

### ICA — Señales ECG (MIT-BIH P-Wave)

```bash
python scripts/ica_ecg.py --seed 42
```

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
- Para resultados idénticos, mantenga constantes: semilla, versiones, datasets.

## Correspondencia con el informe

| Sección del informe | Script | Resumen generado |
|---------------------|--------|------------------|
| 2.1 SVD             | `svd_movielens.py` | `outputs/svd/resumen_svd.md` |
| 2.2 t-SNE           | `tsne_umap_cancer.py` | `outputs/tsne/resumen_tsne.md` |
| 2.3 UMAP            | `tsne_umap_cancer.py` | `outputs/umap/resumen_umap.md` |
| 2.4 ICA             | `ica_ecg.py` | `outputs/ica/resumen_ica.md` |
| 3 Comparación       | `tsne_umap_cancer.py` | `fig_comparison_tsne_vs_umap.png` |
