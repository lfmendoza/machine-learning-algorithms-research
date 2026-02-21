# Resumen ICA - MIT-BIH Arrhythmia P-Wave

## Dataset
- Fuente: MIT-BIH Arrhythmia Database P-Wave Annotations (PhysioNet)
- Registros analizados: ['100', '119', '207']
- Canales por registro: 2 (MLII + V1/V2/V5)
- Frecuencia de muestreo: 360 Hz
- Ventana analizada: 3600 muestras (10.0 s)

## Preprocesamiento
- StandardScaler por canal (media=0, std=1)
- Ventana de 10 segundos desde el inicio del registro

## Método
- FastICA (scikit-learn), 2 componentes, whitening='unit-variance'

## Resultados clave
- Kurtosis promedio (|valor|) — originales: 13.16, ICA: 13.38
- Componentes ICA tienden a mayor kurtosis (mayor no-gaussianidad)

## Figuras generadas
- fig_ica_01_originales_100: Señales originales
- fig_ica_02_componentes_100: Componentes ICA
- fig_ica_03_comparacion_100: Original vs ICA
- fig_ica_01_originales_119: Señales originales
- fig_ica_02_componentes_119: Componentes ICA
- fig_ica_03_comparacion_119: Original vs ICA
- fig_ica_01_originales_207: Señales originales
- fig_ica_02_componentes_207: Componentes ICA
- fig_ica_03_comparacion_207: Original vs ICA
- fig_ica_04_kurtosis: Kurtosis comparativa
- fig_ica_05_pwave_*: Detalle con anotaciones P-wave

## Tablas generadas
- ica_kurtosis.csv
- ica_mixing_matrix_*.csv (por registro)
