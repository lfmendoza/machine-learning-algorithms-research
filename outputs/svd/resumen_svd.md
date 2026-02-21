# SVD (Descomposición en Valores Singulares)

## 1. Descripción teórica

### Explicación del algoritmo y objetivo principal

La Descomposición en Valores Singulares (SVD) factoriza una matriz A de dimensiones m×n en tres matrices: A = U·Σ·Vᵀ, donde U (m×m) contiene los vectores singulares izquierdos, Σ (m×n) es una matriz diagonal con los valores singulares ordenados de mayor a menor, y Vᵀ (n×n) contiene los vectores singulares derechos. Su objetivo principal es la reducción de dimensionalidad: al retener solo los k valores singulares más grandes se obtiene la mejor aproximación de rango k en norma de Frobenius.

### Principales características y supuestos

- Es un método determinístico y algebraicamente exacto (no iterativo en su forma teórica).
- No requiere que los datos sigan una distribución particular (no asume normalidad).
- Opera directamente sobre la matriz de datos, sin necesidad de calcular la matriz de covarianza.
- La variante TruncatedSVD utilizada aquí trabaja eficientemente con matrices dispersas, ya que no centra los datos (no resta la media), a diferencia de PCA.
- Los valores singulares reflejan la importancia relativa de cada componente: σ₁ ≥ σ₂ ≥ ... ≥ σₖ.

### Diferencias con PCA

| Aspecto | SVD (TruncatedSVD) | PCA |
|---|---|---|
| Centrado | No centra los datos | Centra (resta la media) |
| Matrices dispersas | Soporte nativo (no densifica) | Requiere densificar o usar variantes |
| Base matemática | Factorización directa A = UΣVᵀ | Diagonalización de la covarianza Cov = VΛVᵀ |
| Interpretación | Factores latentes de la matriz original | Direcciones de máxima varianza centrada |
| Caso especial | PCA es SVD aplicado a datos centrados | — |

## 2. Usos y aplicaciones

### Principales usos en análisis de datos

- **Reducción de dimensionalidad**: comprimir datos de alta dimensión preservando la mayor varianza posible.
- **Sistemas de recomendación**: factorización de matrices usuario-ítem para descubrir factores latentes (gustos, categorías implícitas).
- **Compresión de datos e imágenes**: aproximaciones de bajo rango para almacenamiento eficiente.
- **Procesamiento de lenguaje natural (LSA/LSI)**: reducir la matriz término-documento para capturar relaciones semánticas.

### Áreas de aplicación

1. **Sistemas de recomendación (Netflix, Spotify)**: SVD identifica factores latentes en matrices de ratings para predecir preferencias no observadas. Es la base del filtrado colaborativo matricial.
2. **Procesamiento de imágenes y visión por computadora**: la aproximación de bajo rango permite comprimir imágenes reteniendo las estructuras visuales más relevantes, y se usa en reconocimiento facial (eigenfaces).
3. **Bioinformática**: análisis de matrices de expresión génica para identificar patrones de co-expresión entre genes y condiciones experimentales.

## 3. Aplicación práctica

### Dataset utilizado

- **Fuente**: MovieLens 100k (GroupLens Research, University of Minnesota)
- **Usuarios**: 943
- **Películas**: 1682
- **Ratings totales**: 100,000
- **Escala de ratings**: 1 a 5 (enteros)
- **Densidad de la matriz**: 6.30% (altamente dispersa)

### Decisiones de preprocesamiento

- Se construyó una matriz dispersa usuario-película en formato CSR (Compressed Sparse Row) de 943×1682.
- Se utilizaron los ratings directos como valores (sin centrar), apropiado para TruncatedSVD sobre matrices dispersas.
- Se solicitaron 50 componentes para el análisis.

### Resultados obtenidos

- **Componentes utilizados**: 50
- **Varianza explicada por el 1er componente**: 15.39%
- **Varianza acumulada (primeros 5 componentes)**: 28.59%
- **Varianza acumulada total (50 componentes)**: 52.35%
- **Componentes necesarios para 80% de varianza**: >50 (no alcanzado con 50 componentes)
- **Componentes necesarios para 90% de varianza**: >50 (no alcanzado con 50 componentes)

### Interpretación

Los primeros componentes capturan los patrones de rating más globales (e.g., películas populares universalmente bien calificadas), mientras que los componentes posteriores capturan preferencias más específicas de nichos o géneros. La proyección 2D de películas (fig_svd_03) muestra agrupamientos por género, lo que confirma que SVD descubre factores latentes con interpretación semántica. El error de reconstrucción (fig_svd_04) decrece rápidamente con los primeros componentes, indicando que la información esencial de la matriz se concentra en pocas dimensiones. La tabla de top películas por componente revela qué títulos dominan cada factor latente.

La varianza acumulada con 50 componentes alcanza solo el 52.35%, lo cual es esperado dado que la matriz usuario-película es altamente dispersa (densidad ~6.3%) y contiene mucha variabilidad individual. Esto implica que se necesitarían muchos más componentes para capturar la mayoría de la varianza, pero los primeros componentes ya contienen los patrones más informativos para recomendación.

### Limitaciones

- SVD asume una relación lineal entre los factores latentes; no captura interacciones no lineales en las preferencias de los usuarios.
- La matriz de ratings tiene valores faltantes (celdas vacías = no calificado), que TruncatedSVD trata como ceros; esto puede sesgar los factores hacia películas populares con más ratings.
- No considera información temporal: las preferencias de los usuarios pueden cambiar con el tiempo.
- La interpretación de los factores latentes es subjetiva; no siempre corresponden a conceptos claros como géneros.
- Con matrices muy dispersas como esta (~6% de densidad), la varianza explicada crece lentamente con el número de componentes.

### Figuras generadas

| Figura | Descripción |
|---|---|
| fig_svd_01 | Varianza explicada por componente y acumulada |
| fig_svd_02 | Usuarios proyectados en espacio latente 2D |
| fig_svd_03 | Películas proyectadas en 2D, coloreadas por género |
| fig_svd_04 | Error relativo de reconstrucción vs. componentes |

### Tablas generadas

| Tabla | Contenido |
|---|---|
| svd_varianza_explicada.csv | Varianza explicada y acumulada por componente |
| svd_reconstruccion_error.csv | Error de reconstrucción para distintos k |
| svd_top_peliculas_por_componente.csv | Top 10 películas con mayor peso por factor latente |
