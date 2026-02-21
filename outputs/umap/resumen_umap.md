# UMAP (Uniform Manifold Approximation and Projection)

## 1. Descripción teórica

### Explicación del algoritmo y objetivo principal

UMAP es una técnica de reducción de dimensionalidad no lineal fundamentada en topología algebraica y geometría riemanniana. El algoritmo modela la estructura de alta dimensión como un grafo ponderado de vecinos (fuzzy simplicial set) y luego optimiza un layout en baja dimensión que preserve la topología de ese grafo. En concreto: (1) construye un grafo de k-vecinos más cercanos con pesos exponenciales basados en distancias locales; (2) simetriza el grafo para obtener una representación topológica fuzzy; (3) minimiza la entropía cruzada entre el grafo original y el grafo en el espacio de baja dimensión mediante descenso de gradiente estocástico.

### Principales características y supuestos

- Asume que los datos están distribuidos uniformemente sobre un manifold (variedad) localmente conexo inmerso en el espacio de alta dimensión.
- Preserva tanto la **estructura local** como la **estructura global** de los datos (mejor que t-SNE en este aspecto).
- **n_neighbors** controla el tamaño del vecindario local: valores pequeños enfatizan detalles locales, valores grandes capturan más estructura global.
- **min_dist** controla qué tan compactos son los clusters en la proyección: valores pequeños permiten puntos más apretados, valores grandes los dispersan.
- Es significativamente más **rápido** que t-SNE para datasets grandes (complejidad aproximada O(n^1.14)).
- A diferencia de t-SNE, UMAP puede generar transformaciones para datos nuevos (método `.transform()`).

### Diferencias con PCA y t-SNE

| Aspecto | UMAP | PCA | t-SNE |
|---|---|---|---|
| Transformación | No lineal | Lineal | No lineal |
| Estructura preservada | Local + global | Global (varianza) | Principalmente local |
| Escalabilidad | Buena (O(n^1.14)) | Excelente | Limitada (O(n²)) |
| Datos nuevos | Soporta `.transform()` | Soporta | No soporta |
| Fundamento teórico | Topología algebraica | Álgebra lineal | Teoría de la información |
| Distancias entre clusters | Más interpretables | Interpretables | No interpretables |

## 2. Usos y aplicaciones

### Principales usos en análisis de datos

- **Visualización de datos de alta dimensión**: alternativa más rápida y con mejor preservación global que t-SNE.
- **Preprocesamiento para clustering**: las proyecciones UMAP pueden usarse como entrada para algoritmos de clustering (HDBSCAN, KMeans) mejorando la separación.
- **Exploración de embeddings**: visualización de representaciones de redes neuronales, word embeddings, y features aprendidas.

### Áreas de aplicación

1. **Genómica y single-cell analysis**: UMAP ha reemplazado parcialmente a t-SNE como estándar de visualización en transcriptómica de célula única gracias a su velocidad y mejor preservación de la estructura global entre tipos celulares.
2. **Detección de anomalías en ciberseguridad**: proyección de vectores de características de tráfico de red para identificar visualmente comportamientos anómalos y ataques que se separan de los patrones normales.
3. **Investigación farmacéutica**: visualización de espacios químicos de alta dimensión (descriptores moleculares) para identificar familias de compuestos y candidatos a fármacos.

## 3. Aplicación práctica

### Dataset utilizado

- **Fuente**: Breast Cancer Wisconsin (Diagnostic), UCI / Kaggle
- **Muestras**: 569
- **Features**: 30 (10 medidas × 3 estadísticos: media, error estándar, peor valor)
- **Etiquetas**: Maligno (M) / Benigno (B)

### Decisiones de preprocesamiento

- Mismo preprocesamiento que t-SNE: eliminación de `id` y `diagnosis`, seguido de `StandardScaler`.
- Esto permite una comparación justa entre ambos métodos.

### Parámetros explorados

| Parámetro | Valores |
|---|---|
| n_neighbors | [5, 15, 30, 50] |
| min_dist | [0.1, 0.5] |
| n_components | 2 |

### Resultados obtenidos

- n_neighbors=5, min_dist=0.1: silhouette=0.515, tiempo=23.3s
- n_neighbors=5, min_dist=0.5: silhouette=0.429, tiempo=1.3s
- n_neighbors=15, min_dist=0.1: silhouette=0.448, tiempo=1.9s
- n_neighbors=15, min_dist=0.5: silhouette=0.424, tiempo=1.9s
- n_neighbors=30, min_dist=0.1: silhouette=0.493, tiempo=2.3s
- n_neighbors=30, min_dist=0.5: silhouette=0.446, tiempo=2.3s
- n_neighbors=50, min_dist=0.1: silhouette=0.445, tiempo=2.8s
- n_neighbors=50, min_dist=0.5: silhouette=0.460, tiempo=3.7s

**Mejor configuración**: n_neighbors=5, min_dist=0.1 con silhouette=0.515

### Interpretación

UMAP produce una separación clara entre tumores malignos y benignos con la mejor configuración (n_neighbors=5, min_dist=0.1, silhouette=0.515). El parámetro n_neighbors tiene el mayor impacto: valores pequeños (5) generan clusters más fragmentados con estructura local detallada, mientras que valores grandes (50) producen proyecciones más suaves que capturan la separación global. El min_dist controla la compacidad visual: con min_dist=0.1 los puntos se agrupan densamente, con min_dist=0.5 se dispersan más. Comparado con t-SNE, UMAP tiende a mantener mejor las distancias relativas entre clusters (no solo dentro de ellos), haciendo que la separación espacial entre los grupos M y B sea más interpretable.

### Figuras generadas

| Figura | Descripción |
|---|---|
| fig_umap_01 | Comparación de n_neighbors (grid 2×2, min_dist=0.1) |
| fig_umap_02 | Efecto de min_dist (n_neighbors=15) |
| fig_umap_03 | Mejor proyección individual con leyenda |
| fig_comparison_tsne_vs_umap | Comparación lado a lado con t-SNE |

### Tablas generadas

| Tabla | Contenido |
|---|---|
| umap_params_silhouette.csv | Silhouette y tiempo por configuración |
| umap_best_coords.csv | Coordenadas 2D de la mejor proyección |
