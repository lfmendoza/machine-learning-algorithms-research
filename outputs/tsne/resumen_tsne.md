# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## 1. Descripción teórica

### Explicación del algoritmo y objetivo principal

t-SNE es una técnica de reducción de dimensionalidad no lineal diseñada específicamente para la visualización de datos de alta dimensión en 2D o 3D. El algoritmo funciona en dos etapas: primero, construye una distribución de probabilidad conjunta sobre pares de puntos en el espacio original, de modo que puntos similares tengan alta probabilidad de ser seleccionados como vecinos; segundo, define una distribución t de Student similar en el espacio de baja dimensión y minimiza la divergencia de Kullback-Leibler (KL) entre ambas distribuciones mediante descenso de gradiente.

### Principales características y supuestos

- Preserva la **estructura local**: puntos cercanos en alta dimensión permanecen cercanos en la proyección.
- Utiliza una distribución t de Student (colas pesadas) en el espacio de baja dimensión para aliviar el problema del "crowding" (aglomeración).
- El parámetro **perplexity** controla el balance entre estructura local y global: valores bajos enfatizan vecindarios pequeños, valores altos consideran más contexto.
- Es **no paramétrico**: no asume distribución ni linealidad en los datos.
- Es **estocástico**: distintas ejecuciones pueden dar resultados ligeramente diferentes.
- Las distancias absolutas en la proyección no tienen significado cuantitativo; solo la estructura relativa (vecindades) es interpretable.

### Diferencias con PCA y otros métodos

| Aspecto | t-SNE | PCA |
|---|---|---|
| Tipo de transformación | No lineal | Lineal |
| Preserva | Estructura local (vecindarios) | Varianza global |
| Escalabilidad | O(n²) — costoso para datasets grandes | O(np min(n,p)) — eficiente |
| Determinismo | Estocástico | Determinístico |
| Inversibilidad | No invertible | Invertible |
| Uso principal | Visualización 2D/3D | Reducción de dimensionalidad general |

## 2. Usos y aplicaciones

### Principales usos en análisis de datos

- **Visualización exploratoria**: proyectar datos multidimensionales a 2D para identificar clusters, outliers y patrones que no son evidentes en el espacio original.
- **Validación de clusters**: verificar visualmente si los grupos encontrados por algoritmos de clustering son coherentes.
- **Análisis de embeddings**: visualizar representaciones aprendidas por redes neuronales (word2vec, BERT, etc.).

### Áreas de aplicación

1. **Bioinformática y genómica**: visualización de datos de single-cell RNA-seq para identificar tipos celulares. t-SNE es estándar en herramientas como Seurat y Scanpy para revelar subpoblaciones celulares en miles de dimensiones génicas.
2. **Diagnóstico médico por imágenes**: proyección de features extraídas de imágenes médicas (mamografías, histopatología) para visualizar la separación entre clases benignas y malignas, como en este ejercicio.
3. **Seguridad informática**: visualización de tráfico de red multidimensional para detectar patrones anómalos de intrusión o malware.

## 3. Aplicación práctica

### Dataset utilizado

- **Fuente**: Breast Cancer Wisconsin (Diagnostic), UCI / Kaggle
- **Muestras**: 569 (tumores de mama)
- **Features**: 30 características numéricas (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría, dimensión fractal — cada una con media, error estándar y peor valor)
- **Etiquetas**: Maligno (M) / Benigno (B)

### Decisiones de preprocesamiento

- Se eliminaron las columnas `id` y `diagnosis` (esta última se conservó como etiqueta para colorear).
- Se aplicó `StandardScaler` (media=0, desviación=1) a todas las features, necesario porque t-SNE es sensible a la escala de las variables.

### Parámetros explorados

| Parámetro | Valores |
|---|---|
| Perplexity | [5, 15, 30, 50] |
| Iteraciones | 1000 |
| Inicialización | PCA |
| Learning rate | auto |

### Resultados obtenidos

- Perplexity=5: silhouette=0.425, KL=1.0961, tiempo=8.5s
- Perplexity=15: silhouette=0.489, KL=1.0836, tiempo=4.8s
- Perplexity=30: silhouette=0.466, KL=0.9532, tiempo=5.4s
- Perplexity=50: silhouette=0.515, KL=0.8046, tiempo=6.5s

**Mejor configuración**: perplexity=50 con silhouette=0.515

### Interpretación

La proyección t-SNE logra una separación visual clara entre tumores malignos y benignos. La mejor perplexity (50) produce clusters compactos y bien separados (silhouette=0.515). Perplexidades bajas (5) tienden a fragmentar los clusters en sub-grupos pequeños, mientras que valores altos producen proyecciones más globales pero menos definidas localmente. La divergencia KL baja (0.8046) indica que la distribución en 2D reproduce fielmente las relaciones de vecindad del espacio original. Es importante recordar que las distancias entre clusters en t-SNE no son directamente comparables; solo la cohesión interna de cada grupo es interpretable.

### Figuras generadas

| Figura | Descripción |
|---|---|
| fig_tsne_01 | Comparación de 4 perplexidades (grid 2×2) |
| fig_tsne_02 | Mejor proyección individual con leyenda |

### Tablas generadas

| Tabla | Contenido |
|---|---|
| tsne_params_silhouette.csv | Silhouette, KL divergence y tiempo por perplexity |
| tsne_best_coords.csv | Coordenadas 2D de la mejor proyección |
