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

Cada algoritmo se aplica a un dataset específico según los requerimientos del laboratorio. Los reportes detallados con resultados numéricos, figuras y tablas se generan automáticamente al ejecutar los scripts y se encuentran en `outputs/<algoritmo>/resumen_<algoritmo>.md`.

### 2.1 SVD (Descomposición en Valores Singulares)

**Descripción teórica**: SVD factoriza una matriz A de dimensiones m×n en tres matrices: A = U·Σ·Vᵀ, donde U contiene los vectores singulares izquierdos, Σ es diagonal con los valores singulares ordenados de mayor a menor, y Vᵀ contiene los vectores singulares derechos. Su objetivo es la reducción de dimensionalidad: al retener solo los k valores singulares más grandes se obtiene la mejor aproximación de rango k. A diferencia de PCA, TruncatedSVD no centra los datos, lo que le permite trabajar eficientemente con matrices dispersas.

**Usos y aplicaciones**: sistemas de recomendación (factorización de matrices usuario-ítem), compresión de imágenes (aproximación de bajo rango), y análisis semántico latente (LSA) en procesamiento de lenguaje natural.

**Dataset utilizado**: MovieLens 100k — 943 usuarios, 1682 películas, 100,000 ratings (escala 1-5). Se construyó una matriz dispersa CSR de 943×1682 con densidad ~6.3%.

**Resultados**: se extrajeron 50 componentes; el primer componente explica ~15% de la varianza. Las proyecciones 2D de películas muestran agrupamientos por género. El error de reconstrucción decrece rápidamente con los primeros componentes.

**Interpretación**: los primeros factores capturan patrones de rating globales (películas universalmente populares), mientras que los posteriores capturan preferencias de nicho. La baja varianza acumulada (~49% con 50 componentes) refleja la alta dispersión de la matriz. Reporte completo: `outputs/svd/resumen_svd.md`.

### 2.2 t-SNE (Incrustación Estocástica de Vecinos con Distribución t)

**Descripción teórica**: t-SNE construye distribuciones de probabilidad sobre pares de puntos en alta dimensión (gaussiana) y baja dimensión (t de Student), y minimiza la divergencia KL entre ambas. Preserva la estructura local: puntos cercanos en el espacio original permanecen cercanos en la proyección. Es no lineal, estocástico y no invertible, a diferencia de PCA.

**Usos y aplicaciones**: visualización exploratoria de datos de alta dimensión, validación visual de agrupamientos, y análisis de representaciones de redes neuronales. Se usa extensamente en bioinformática (RNA-seq de célula única) y diagnóstico médico por imágenes.

**Dataset utilizado**: Breast Cancer Wisconsin (Diagnostic) — 569 muestras de tumores de mama con 30 características numéricas (radio, textura, perímetro, etc.) y etiqueta M/B. Se aplicó StandardScaler previo.

**Resultados**: se exploraron perplejidades {5, 15, 30, 50}. La mejor configuración (perplejidad=50) alcanzó una silueta de 0.515 con divergencia KL de 0.80. La proyección separa claramente tumores malignos de benignos.

**Interpretación**: perplejidades bajas fragmentan los grupos, mientras que valores altos producen visualizaciones más globales. Las distancias entre grupos en t-SNE no son interpretables; solo la cohesión intra-grupo es confiable. Reporte completo: `outputs/tsne/resumen_tsne.md`.

### 2.3 UMAP (Aproximación y Proyección Uniforme de Variedades)

**Descripción teórica**: UMAP modela los datos como un grafo ponderado de k-vecinos (conjunto simplicial difuso) y optimiza un layout en baja dimensión que preserve la topología del grafo. Se basa en topología algebraica y geometría riemanniana. A diferencia de t-SNE, preserva mejor la estructura global, es más rápido (O(n^1.14) vs O(n²)) y permite transformar datos nuevos.

**Usos y aplicaciones**: alternativa rápida a t-SNE para visualización, preprocesamiento para algoritmos de agrupamiento (HDBSCAN, KMeans), y exploración de espacios químicos en investigación farmacéutica.

**Dataset utilizado**: Breast Cancer Wisconsin (mismo dataset que t-SNE para comparación directa).

**Resultados**: se exploraron n_neighbors {5, 15, 30, 50} × min_dist {0.1, 0.5}. La mejor configuración (n_neighbors=5, min_dist=0.1) alcanzó una silueta de 0.515. La comparación lado a lado con t-SNE muestra que UMAP mantiene mejor las distancias relativas entre grupos.

**Interpretación**: n_neighbors controla el balance local/global; min_dist controla la compacidad visual. UMAP produce grupos con distancias inter-grupo más interpretables que t-SNE. Reporte completo: `outputs/umap/resumen_umap.md`.

### 2.4 ICA (Análisis de Componentes Independientes)

**Descripción teórica**: ICA es una técnica de separación ciega de fuentes que recupera señales independientes a partir de mezclas lineales desconocidas. Dado X = A·S, estima W ≈ A⁻¹ tal que S ≈ W·X, maximizando la no-gaussianidad (medida por kurtosis o negentropía). A diferencia de PCA, que solo decorrelaciona (orden 2), ICA capta dependencias de orden superior para lograr independencia estadística.

**Usos y aplicaciones**: separación de fuentes en señales ECG/EEG (eliminar artefactos musculares u oculares), procesamiento de audio (problema del cóctel), y extracción de características independientes para clasificación.

**Dataset utilizado**: MIT-BIH Arrhythmia Database P-Wave Annotations (PhysioNet) — 12 registros ECG de 2 canales (MLII + V1/V2/V5) a 360 Hz. Se analizaron registros 100, 119 y 207 con ventanas de 10 segundos.

**Resultados**: FastICA separó 2 componentes independientes por registro. La kurtosis promedio de los componentes ICA (13.38) supera la de los canales originales (13.16), confirmando la maximización de no-gaussianidad. Las anotaciones de onda P coinciden con morfologías recurrentes en las componentes separadas.

**Interpretación**: un componente tiende a capturar la actividad ventricular dominante (QRS), mientras que el otro aísla mejor las ondas P y T. La matriz de mezcla estimada revela la contribución de cada derivación. Limitaciones: solo 2 canales limitan la separación; el modelo asume mezcla lineal instantánea. Reporte completo: `outputs/ica/resumen_ica.md`.

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
