# t-SNE (Incrustación Estocástica de Vecinos con Distribución t)

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

- **Visualización exploratoria**: proyectar datos multidimensionales a 2D para identificar agrupamientos, valores atípicos y patrones que no son evidentes en el espacio original.
- **Validación de agrupamientos**: verificar visualmente si los grupos encontrados por algoritmos de agrupamiento son coherentes.
- **Análisis de representaciones vectoriales**: visualizar representaciones aprendidas por redes neuronales (word2vec, BERT, etc.).

### Áreas de aplicación

1. **Bioinformática y genómica**: visualización de datos de RNA-seq de célula única para identificar tipos celulares. t-SNE es estándar en herramientas como Seurat y Scanpy para revelar subpoblaciones celulares en miles de dimensiones génicas.
2. **Diagnóstico médico por imágenes**: proyección de características extraídas de imágenes médicas (mamografías, histopatología) para visualizar la separación entre clases benignas y malignas, como en este ejercicio.
3. **Seguridad informática**: visualización de tráfico de red multidimensional para detectar patrones anómalos de intrusión o malware.

## 3. Aplicación práctica

### Dataset utilizado

- **Fuente**: Breast Cancer Wisconsin (Diagnostic), UCI / Kaggle
- **Muestras**: 569 (tumores de mama)
- **Características**: 30 variables numéricas (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría, dimensión fractal — cada una con media, error estándar y peor valor)
- **Etiquetas**: Maligno (M) / Benigno (B)

### Decisiones de preprocesamiento

- Se eliminaron las columnas `id` y `diagnosis` (esta última se conservó como etiqueta para colorear).
- Se aplicó `StandardScaler` (media=0, desviación=1) a todas las características, necesario porque t-SNE es sensible a la escala de las variables.

### Parámetros explorados

| Parámetro | Valores |
|---|---|
| Perplejidad (perplexity) | [5, 15, 30, 50] |
| Iteraciones | 1000 |
| Inicialización | PCA |
| Tasa de aprendizaje | auto |

### Resultados obtenidos

- Perplexity=5: silueta=0.425, KL=1.0961, tiempo=5.3s
- Perplexity=15: silueta=0.489, KL=1.0836, tiempo=3.9s
- Perplexity=30: silueta=0.466, KL=0.9532, tiempo=4.0s
- Perplexity=50: silueta=0.515, KL=0.8046, tiempo=6.5s

**Mejor configuración**: perplexity=50 con silueta=0.515

### Interpretación

La proyección t-SNE logra una separación visual clara entre tumores malignos y benignos. La mejor perplejidad (50) produce agrupamientos compactos y bien separados (silueta=0.515). Perplejidades bajas (5) tienden a fragmentar los agrupamientos en sub-grupos pequeños, mientras que valores altos producen proyecciones más globales pero menos definidas localmente. La divergencia KL baja (0.8046) indica que la distribución en 2D reproduce fielmente las relaciones de vecindad del espacio original. Es importante recordar que las distancias entre grupos en t-SNE no son directamente comparables; solo la cohesión interna de cada grupo es interpretable.

### Limitaciones

- **No preserva distancias globales**: las distancias entre grupos separados en la proyección no son interpretables; solo la estructura intra-grupo es confiable.
- **Sensibilidad a la perplejidad**: distintos valores de perplejidad producen visualizaciones muy diferentes, lo que puede llevar a interpretaciones erróneas si no se exploran múltiples configuraciones.
- **No determinístico**: cada ejecución sin semilla fija puede producir proyecciones diferentes, complicando la reproducibilidad.
- **Escalabilidad limitada**: la complejidad O(n²) lo hace impracticable para datasets con más de ~10,000 observaciones sin técnicas de aproximación.
- **No permite proyectar datos nuevos**: a diferencia de PCA o UMAP, no se puede aplicar la transformación aprendida a observaciones fuera del conjunto de entrenamiento.

### Figuras generadas

| Figura | Descripción |
|---|---|
| fig_tsne_01 | Comparación de 4 perplejidades (cuadrícula 2×2) |
| fig_tsne_02 | Mejor proyección individual con leyenda |

### Tablas generadas

| Tabla | Contenido |
|---|---|
| tsne_params_silhouette.csv | Silueta, divergencia KL y tiempo por perplejidad |
| tsne_best_coords.csv | Coordenadas 2D de la mejor proyección |
