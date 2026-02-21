# Resumen SVD - MovieLens 100k

## Dataset
- Fuente: MovieLens 100k (GroupLens Research)
- Usuarios: 943
- Películas: 1682
- Ratings: 100000
- Rango de ratings: 1-5

## Preprocesamiento
- Matriz dispersa usuario-película (CSR)
- Valores: rating directo (sin centrar)

## Parámetros
- Componentes solicitados: 50
- Componentes para 80% varianza: nan
- Componentes para 90% varianza: nan

## Figuras generadas
- fig_svd_01: Varianza explicada por componente y acumulada
- fig_svd_02: Usuarios en espacio latente 2D
- fig_svd_03: Películas en espacio latente 2D por género
- fig_svd_04: Error de reconstrucción vs. número de componentes

## Tablas generadas
- svd_varianza_explicada.csv
- svd_reconstruccion_error.csv
- svd_top_peliculas_por_componente.csv
