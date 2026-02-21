#!/usr/bin/env python3
"""
t-SNE y UMAP aplicados al Breast Cancer Wisconsin (Diagnostic) Dataset.

Genera proyecciones 2D con distintas configuraciones de parámetros,
métricas de separación (silhouette) y una comparación lado a lado.

Uso:
  python scripts/tsne_umap_cancer.py
  python scripts/tsne_umap_cancer.py --seed 42
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import umap


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


LABEL_COLORS = {"M": "#d62728", "B": "#1f77b4"}
LABEL_NAMES = {"M": "Maligno", "B": "Benigno"}


# ─────────────────────────────── Data ───────────────────────────────

def load_breast_cancer(data_path: str):
    df = pd.read_csv(data_path, encoding="utf-8")
    # Limpiar columna residual sin nombre al final
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    labels = df["diagnosis"].values
    feature_df = df.drop(columns=["id", "diagnosis"])
    return feature_df, labels


def preprocess(feature_df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)
    return X


# ─────────────────────────────── t-SNE ──────────────────────────────

def run_tsne_grid(X, labels, output_dir: str, seed: int):
    """Ejecuta t-SNE con distintas perplexidades y genera comparación."""
    _ensure(output_dir)

    perplexities = [5, 15, 30, 50]
    results = {}
    metrics_rows = []

    for perp in perplexities:
        t0 = time.time()
        tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                    init="pca", learning_rate="auto", random_state=seed)
        Z = tsne.fit_transform(X)
        elapsed = time.time() - t0

        le = LabelEncoder().fit(labels)
        sil = silhouette_score(Z, le.transform(labels))
        results[perp] = Z
        metrics_rows.append({
            "perplexity": perp,
            "kl_divergence": float(tsne.kl_divergence_),
            "silhouette": float(sil),
            "tiempo_s": round(elapsed, 2),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(output_dir, "tsne_params_silhouette.csv"),
                      index=False, encoding="utf-8")

    # Grid de perplexidades
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, perp in zip(axes.flat, perplexities):
        Z = results[perp]
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7,
                       color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        sil_val = metrics_df.loc[metrics_df["perplexity"] == perp,
                                 "silhouette"].iloc[0]
        ax.set_title(f"perplejidad = {perp}  (silueta = {sil_val:.3f})", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls="--", lw=0.3)

    fig.suptitle("t-SNE — Efecto de la perplejidad", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_tsne_01_perplexity_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Mejor proyección individual
    best_perp = int(metrics_df.sort_values("silhouette", ascending=False)
                    .iloc[0]["perplexity"])
    Z_best = results[best_perp]
    best_sil = metrics_df.loc[metrics_df["perplexity"] == best_perp,
                              "silhouette"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(Z_best[mask, 0], Z_best[mask, 1], s=14, alpha=0.7,
                   color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
    ax.set_title(f"t-SNE (perplejidad={best_perp}, silueta={best_sil:.3f})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True, ls="--", lw=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_tsne_02_best_projection.png"),
                dpi=200)
    plt.close(fig)

    pd.DataFrame({"tsne_1": Z_best[:, 0], "tsne_2": Z_best[:, 1],
                   "diagnosis": labels}).to_csv(
        os.path.join(output_dir, "tsne_best_coords.csv"),
        index=False, encoding="utf-8")

    return results, best_perp


# ─────────────────────────────── UMAP ───────────────────────────────

def run_umap_grid(X, labels, output_dir: str, seed: int):
    """Ejecuta UMAP con distintos n_neighbors y min_dist."""
    _ensure(output_dir)

    neighbors_list = [5, 15, 30, 50]
    min_dist_list = [0.1, 0.5]
    results = {}
    metrics_rows = []

    for nn in neighbors_list:
        for md in min_dist_list:
            t0 = time.time()
            reducer = umap.UMAP(n_components=2, n_neighbors=nn,
                                min_dist=md, random_state=seed)
            Z = reducer.fit_transform(X)
            elapsed = time.time() - t0

            le = LabelEncoder().fit(labels)
            sil = silhouette_score(Z, le.transform(labels))
            results[(nn, md)] = Z
            metrics_rows.append({
                "n_neighbors": nn,
                "min_dist": md,
                "silhouette": float(sil),
                "tiempo_s": round(elapsed, 2),
            })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(output_dir, "umap_params_silhouette.csv"),
                      index=False, encoding="utf-8")

    # Grid variando n_neighbors (min_dist fijo = 0.1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, nn in zip(axes.flat, neighbors_list):
        Z = results[(nn, 0.1)]
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7,
                       color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        sil_val = metrics_df.loc[(metrics_df["n_neighbors"] == nn) &
                                 (metrics_df["min_dist"] == 0.1),
                                 "silhouette"].iloc[0]
        ax.set_title(f"n_neighbors={nn}, min_dist=0.1 (silueta={sil_val:.3f})",
                     fontsize=9)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls="--", lw=0.3)

    fig.suptitle("UMAP — Efecto de n_neighbors (min_dist=0.1)", fontsize=13,
                 y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_umap_01_neighbors_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Efecto de min_dist (n_neighbors=15)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, md in zip(axes.flat, min_dist_list):
        Z = results[(15, md)]
        for lab in np.unique(labels):
            mask = labels == lab
            ax.scatter(Z[mask, 0], Z[mask, 1], s=14, alpha=0.7,
                       color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        sil_val = metrics_df.loc[(metrics_df["n_neighbors"] == 15) &
                                 (metrics_df["min_dist"] == md),
                                 "silhouette"].iloc[0]
        ax.set_title(f"min_dist={md} (silueta={sil_val:.3f})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", lw=0.3)

    fig.suptitle("UMAP — Efecto de min_dist (n_neighbors=15)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_umap_02_mindist_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Mejor proyección individual
    best_row = metrics_df.sort_values("silhouette", ascending=False).iloc[0]
    best_nn = int(best_row["n_neighbors"])
    best_md = float(best_row["min_dist"])
    best_sil = best_row["silhouette"]
    Z_best = results[(best_nn, best_md)]

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(Z_best[mask, 0], Z_best[mask, 1], s=14, alpha=0.7,
                   color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
    ax.set_title(f"UMAP (n_neighbors={best_nn}, min_dist={best_md}, "
                 f"silueta={best_sil:.3f})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.grid(True, ls="--", lw=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_umap_03_best_projection.png"),
                dpi=200)
    plt.close(fig)

    pd.DataFrame({"umap_1": Z_best[:, 0], "umap_2": Z_best[:, 1],
                   "diagnosis": labels}).to_csv(
        os.path.join(output_dir, "umap_best_coords.csv"),
        index=False, encoding="utf-8")

    return results, (best_nn, best_md)


# ──────────────────────── Comparison figure ─────────────────────────

def plot_comparison(tsne_results, best_perp, umap_results, best_umap_key,
                    labels, output_dir: str) -> None:
    Z_tsne = tsne_results[best_perp]
    Z_umap = umap_results[best_umap_key]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for lab in np.unique(labels):
        mask = labels == lab
        axes[0].scatter(Z_tsne[mask, 0], Z_tsne[mask, 1], s=12, alpha=0.7,
                        color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        axes[1].scatter(Z_umap[mask, 0], Z_umap[mask, 1], s=12, alpha=0.7,
                        color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])

    axes[0].set_title(f"t-SNE (perplexity={best_perp})")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls="--", lw=0.3)

    nn, md = best_umap_key
    axes[1].set_title(f"UMAP (n_neighbors={nn}, min_dist={md})")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", lw=0.3)

    fig.suptitle("Comparación t-SNE vs UMAP — Cáncer de Mama Wisconsin",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_comparison_tsne_vs_umap.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────── Summaries ─────────────────────────────────

def write_tsne_summary(metrics_path: str, output_dir: str,
                       n_samples: int, n_features: int) -> None:
    metrics = pd.read_csv(metrics_path)
    path = os.path.join(output_dir, "resumen_tsne.md")
    best = metrics.sort_values("silhouette", ascending=False).iloc[0]
    worst = metrics.sort_values("silhouette", ascending=True).iloc[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# t-SNE (Incrustación Estocástica de Vecinos con Distribución t)\n\n")

        # ── 1. Descripción teórica ──
        f.write("## 1. Descripción teórica\n\n")
        f.write("### Explicación del algoritmo y objetivo principal\n\n")
        f.write(
            "t-SNE es una técnica de reducción de dimensionalidad no lineal "
            "diseñada específicamente para la visualización de datos de alta "
            "dimensión en 2D o 3D. El algoritmo funciona en dos etapas: "
            "primero, construye una distribución de probabilidad conjunta "
            "sobre pares de puntos en el espacio original, de modo que puntos "
            "similares tengan alta probabilidad de ser seleccionados como "
            "vecinos; segundo, define una distribución t de Student similar "
            "en el espacio de baja dimensión y minimiza la divergencia de "
            "Kullback-Leibler (KL) entre ambas distribuciones mediante "
            "descenso de gradiente.\n\n")
        f.write("### Principales características y supuestos\n\n")
        f.write(
            "- Preserva la **estructura local**: puntos cercanos en alta "
            "dimensión permanecen cercanos en la proyección.\n"
            "- Utiliza una distribución t de Student (colas pesadas) en el "
            "espacio de baja dimensión para aliviar el problema del "
            "\"crowding\" (aglomeración).\n"
            "- El parámetro **perplexity** controla el balance entre "
            "estructura local y global: valores bajos enfatizan vecindarios "
            "pequeños, valores altos consideran más contexto.\n"
            "- Es **no paramétrico**: no asume distribución ni linealidad "
            "en los datos.\n"
            "- Es **estocástico**: distintas ejecuciones pueden dar "
            "resultados ligeramente diferentes.\n"
            "- Las distancias absolutas en la proyección no tienen "
            "significado cuantitativo; solo la estructura relativa "
            "(vecindades) es interpretable.\n\n")
        f.write("### Diferencias con PCA y otros métodos\n\n")
        f.write(
            "| Aspecto | t-SNE | PCA |\n"
            "|---|---|---|\n"
            "| Tipo de transformación | No lineal | Lineal |\n"
            "| Preserva | Estructura local (vecindarios) | Varianza "
            "global |\n"
            "| Escalabilidad | O(n²) — costoso para datasets grandes | "
            "O(np min(n,p)) — eficiente |\n"
            "| Determinismo | Estocástico | Determinístico |\n"
            "| Inversibilidad | No invertible | Invertible |\n"
            "| Uso principal | Visualización 2D/3D | Reducción de "
            "dimensionalidad general |\n\n")

        # ── 2. Usos y aplicaciones ──
        f.write("## 2. Usos y aplicaciones\n\n")
        f.write("### Principales usos en análisis de datos\n\n")
        f.write(
            "- **Visualización exploratoria**: proyectar datos "
            "multidimensionales a 2D para identificar agrupamientos, "
            "valores atípicos y patrones que no son evidentes en el "
            "espacio original.\n"
            "- **Validación de agrupamientos**: verificar visualmente si "
            "los grupos encontrados por algoritmos de agrupamiento son "
            "coherentes.\n"
            "- **Análisis de representaciones vectoriales**: visualizar "
            "representaciones aprendidas por redes neuronales (word2vec, "
            "BERT, etc.).\n\n")
        f.write("### Áreas de aplicación\n\n")
        f.write(
            "1. **Bioinformática y genómica**: visualización de datos de "
            "RNA-seq de célula única para identificar tipos celulares. "
            "t-SNE es estándar en herramientas como Seurat y Scanpy para "
            "revelar subpoblaciones celulares en miles de dimensiones "
            "génicas.\n"
            "2. **Diagnóstico médico por imágenes**: proyección de "
            "características extraídas de imágenes médicas (mamografías, "
            "histopatología) para visualizar la separación entre clases "
            "benignas y malignas, como en este ejercicio.\n"
            "3. **Seguridad informática**: visualización de tráfico de red "
            "multidimensional para detectar patrones anómalos de intrusión "
            "o malware.\n\n")

        # ── 3. Aplicación práctica ──
        f.write("## 3. Aplicación práctica\n\n")
        f.write("### Dataset utilizado\n\n")
        f.write(
            f"- **Fuente**: Breast Cancer Wisconsin (Diagnostic), UCI / "
            f"Kaggle\n"
            f"- **Muestras**: {n_samples} (tumores de mama)\n"
            f"- **Características**: {n_features} variables numéricas "
            f"(radio, textura, perímetro, área, suavidad, compacidad, "
            f"concavidad, puntos cóncavos, simetría, dimensión fractal — "
            f"cada una con media, error estándar y peor valor)\n"
            f"- **Etiquetas**: Maligno (M) / Benigno (B)\n\n")
        f.write("### Decisiones de preprocesamiento\n\n")
        f.write(
            "- Se eliminaron las columnas `id` y `diagnosis` (esta última "
            "se conservó como etiqueta para colorear).\n"
            "- Se aplicó `StandardScaler` (media=0, desviación=1) a todas "
            "las características, necesario porque t-SNE es sensible a la "
            "escala de las variables.\n\n")
        f.write("### Parámetros explorados\n\n")
        f.write(
            f"| Parámetro | Valores |\n"
            f"|---|---|\n"
            f"| Perplejidad (perplexity) | {metrics['perplexity'].tolist()} |\n"
            f"| Iteraciones | 1000 |\n"
            f"| Inicialización | PCA |\n"
            f"| Tasa de aprendizaje | auto |\n\n")
        f.write("### Resultados obtenidos\n\n")
        for _, row in metrics.iterrows():
            f.write(
                f"- Perplexity={int(row['perplexity'])}: "
                f"silueta={row['silhouette']:.3f}, "
                f"KL={row['kl_divergence']:.4f}, "
                f"tiempo={row['tiempo_s']:.1f}s\n")
        f.write(
            f"\n**Mejor configuración**: perplexity={int(best['perplexity'])} "
            f"con silueta={best['silhouette']:.3f}\n\n")
        f.write("### Interpretación\n\n")
        f.write(
            f"La proyección t-SNE logra una separación visual clara entre "
            f"tumores malignos y benignos. La mejor perplejidad "
            f"({int(best['perplexity'])}) produce agrupamientos compactos y "
            f"bien separados (silueta={best['silhouette']:.3f}). "
            f"Perplejidades bajas ({int(worst['perplexity'])}) tienden a "
            f"fragmentar los agrupamientos en sub-grupos pequeños, mientras "
            f"que valores altos producen proyecciones más globales pero "
            f"menos definidas localmente. La divergencia KL baja "
            f"({best['kl_divergence']:.4f}) indica que la distribución en "
            f"2D reproduce fielmente las relaciones de vecindad del espacio "
            f"original. Es importante recordar que las distancias entre "
            f"grupos en t-SNE no son directamente comparables; solo la "
            f"cohesión interna de cada grupo es interpretable.\n\n")
        f.write("### Figuras generadas\n\n")
        f.write(
            "| Figura | Descripción |\n"
            "|---|---|\n"
            "| fig_tsne_01 | Comparación de 4 perplexidades (grid 2×2) |\n"
            "| fig_tsne_02 | Mejor proyección individual con leyenda |\n\n")
        f.write("### Tablas generadas\n\n")
        f.write(
            "| Tabla | Contenido |\n"
            "|---|---|\n"
            "| tsne_params_silhouette.csv | Silueta, divergencia KL y "
            "tiempo por perplejidad |\n"
            "| tsne_best_coords.csv | Coordenadas 2D de la mejor "
            "proyección |\n")


def write_umap_summary(metrics_path: str, output_dir: str,
                       n_samples: int, n_features: int) -> None:
    metrics = pd.read_csv(metrics_path)
    path = os.path.join(output_dir, "resumen_umap.md")
    best = metrics.sort_values("silhouette", ascending=False).iloc[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# UMAP (Aproximación y Proyección Uniforme de Variedades)\n\n")

        # ── 1. Descripción teórica ──
        f.write("## 1. Descripción teórica\n\n")
        f.write("### Explicación del algoritmo y objetivo principal\n\n")
        f.write(
            "UMAP es una técnica de reducción de dimensionalidad no lineal "
            "fundamentada en topología algebraica y geometría riemanniana. "
            "El algoritmo modela la estructura de alta dimensión como un "
            "grafo ponderado de vecinos (fuzzy simplicial set) y luego "
            "optimiza un layout en baja dimensión que preserve la topología "
            "de ese grafo. En concreto: (1) construye un grafo de k-vecinos "
            "más cercanos con pesos exponenciales basados en distancias "
            "locales; (2) simetriza el grafo para obtener una representación "
            "topológica fuzzy; (3) minimiza la entropía cruzada entre el "
            "grafo original y el grafo en el espacio de baja dimensión "
            "mediante descenso de gradiente estocástico.\n\n")
        f.write("### Principales características y supuestos\n\n")
        f.write(
            "- Asume que los datos están distribuidos uniformemente sobre un "
            "manifold (variedad) localmente conexo inmerso en el espacio de "
            "alta dimensión.\n"
            "- Preserva tanto la **estructura local** como la **estructura "
            "global** de los datos (mejor que t-SNE en este aspecto).\n"
            "- **n_neighbors** controla el tamaño del vecindario local: "
            "valores pequeños enfatizan detalles locales, valores grandes "
            "capturan más estructura global.\n"
            "- **min_dist** controla qué tan compactos son los clusters en "
            "la proyección: valores pequeños permiten puntos más apretados, "
            "valores grandes los dispersan.\n"
            "- Es significativamente más **rápido** que t-SNE para datasets "
            "grandes (complejidad aproximada O(n^1.14)).\n"
            "- A diferencia de t-SNE, UMAP puede generar transformaciones "
            "para datos nuevos (método `.transform()`).\n\n")
        f.write("### Diferencias con PCA y t-SNE\n\n")
        f.write(
            "| Aspecto | UMAP | PCA | t-SNE |\n"
            "|---|---|---|---|\n"
            "| Transformación | No lineal | Lineal | No lineal |\n"
            "| Estructura preservada | Local + global | Global (varianza) "
            "| Principalmente local |\n"
            "| Escalabilidad | Buena (O(n^1.14)) | Excelente | "
            "Limitada (O(n²)) |\n"
            "| Datos nuevos | Soporta `.transform()` | Soporta | "
            "No soporta |\n"
            "| Fundamento teórico | Topología algebraica | Álgebra lineal | "
            "Teoría de la información |\n"
            "| Distancias entre clusters | Más interpretables | "
            "Interpretables | No interpretables |\n\n")

        # ── 2. Usos y aplicaciones ──
        f.write("## 2. Usos y aplicaciones\n\n")
        f.write("### Principales usos en análisis de datos\n\n")
        f.write(
            "- **Visualización de datos de alta dimensión**: alternativa "
            "más rápida y con mejor preservación global que t-SNE.\n"
            "- **Preprocesamiento para agrupamiento**: las proyecciones "
            "UMAP pueden usarse como entrada para algoritmos de "
            "agrupamiento (HDBSCAN, KMeans) mejorando la separación.\n"
            "- **Exploración de representaciones vectoriales**: "
            "visualización de representaciones de redes neuronales, "
            "vectores de palabras y características aprendidas.\n\n")
        f.write("### Áreas de aplicación\n\n")
        f.write(
            "1. **Genómica y análisis de célula única**: UMAP ha "
            "reemplazado parcialmente a t-SNE como estándar de "
            "visualización en transcriptómica de célula única gracias a "
            "su velocidad y mejor preservación de la estructura global "
            "entre tipos celulares.\n"
            "2. **Detección de anomalías en ciberseguridad**: proyección "
            "de vectores de características de tráfico de red para "
            "identificar visualmente comportamientos anómalos y ataques "
            "que se separan de los patrones normales.\n"
            "3. **Investigación farmacéutica**: visualización de espacios "
            "químicos de alta dimensión (descriptores moleculares) para "
            "identificar familias de compuestos y candidatos a "
            "fármacos.\n\n")

        # ── 3. Aplicación práctica ──
        f.write("## 3. Aplicación práctica\n\n")
        f.write("### Dataset utilizado\n\n")
        f.write(
            f"- **Fuente**: Breast Cancer Wisconsin (Diagnostic), UCI / "
            f"Kaggle\n"
            f"- **Muestras**: {n_samples}\n"
            f"- **Características**: {n_features} (10 medidas × 3 "
            f"estadísticos: media, error estándar, peor valor)\n"
            f"- **Etiquetas**: Maligno (M) / Benigno (B)\n\n")
        f.write("### Decisiones de preprocesamiento\n\n")
        f.write(
            "- Mismo preprocesamiento que t-SNE: eliminación de `id` y "
            "`diagnosis`, seguido de `StandardScaler`.\n"
            "- Esto permite una comparación justa entre ambos métodos.\n\n")
        f.write("### Parámetros explorados\n\n")
        nn_list = sorted(metrics['n_neighbors'].unique().tolist())
        md_list = sorted(metrics['min_dist'].unique().tolist())
        f.write(
            f"| Parámetro | Valores |\n"
            f"|---|---|\n"
            f"| n_neighbors | {nn_list} |\n"
            f"| min_dist | {md_list} |\n"
            f"| n_components | 2 |\n\n")
        f.write("### Resultados obtenidos\n\n")
        for _, row in metrics.iterrows():
            f.write(
                f"- n_neighbors={int(row['n_neighbors'])}, "
                f"min_dist={row['min_dist']}: "
                f"silueta={row['silhouette']:.3f}, "
                f"tiempo={row['tiempo_s']:.1f}s\n")
        f.write(
            f"\n**Mejor configuración**: n_neighbors="
            f"{int(best['n_neighbors'])}, min_dist={best['min_dist']} "
            f"con silueta={best['silhouette']:.3f}\n\n")
        f.write("### Interpretación\n\n")
        f.write(
            f"UMAP produce una separación clara entre tumores malignos y "
            f"benignos con la mejor configuración (n_neighbors="
            f"{int(best['n_neighbors'])}, min_dist={best['min_dist']}, "
            f"silueta={best['silhouette']:.3f}). "
            f"El parámetro n_neighbors tiene el mayor impacto: valores "
            f"pequeños (5) generan agrupamientos más fragmentados con "
            f"estructura local detallada, mientras que valores grandes (50) "
            f"producen proyecciones más suaves que capturan la separación "
            f"global. El min_dist controla la compacidad visual: con "
            f"min_dist=0.1 los puntos se agrupan densamente, con "
            f"min_dist=0.5 se dispersan más. Comparado con t-SNE, UMAP "
            f"tiende a mantener mejor las distancias relativas entre "
            f"grupos (no solo dentro de ellos), haciendo que la separación "
            f"espacial entre los grupos M y B sea más "
            f"interpretable.\n\n")
        f.write("### Figuras generadas\n\n")
        f.write(
            "| Figura | Descripción |\n"
            "|---|---|\n"
            "| fig_umap_01 | Comparación de n_neighbors (grid 2×2, "
            "min_dist=0.1) |\n"
            "| fig_umap_02 | Efecto de min_dist (n_neighbors=15) |\n"
            "| fig_umap_03 | Mejor proyección individual con leyenda |\n"
            "| fig_comparison_tsne_vs_umap | Comparación lado a lado con "
            "t-SNE |\n\n")
        f.write("### Tablas generadas\n\n")
        f.write(
            "| Tabla | Contenido |\n"
            "|---|---|\n"
            "| umap_params_silhouette.csv | Silueta y tiempo por "
            "configuración |\n"
            "| umap_best_coords.csv | Coordenadas 2D de la mejor "
            "proyección |\n")


# ─────────────────────────────── Main ───────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="t-SNE y UMAP sobre Breast Cancer Wisconsin.")
    parser.add_argument("--data",
                        default=os.path.join("datasets", "brest-cancer",
                                             "data.csv"),
                        help="Ruta al CSV de Breast Cancer.")
    parser.add_argument("--output-tsne",
                        default=os.path.join("outputs", "tsne"),
                        help="Directorio de salida t-SNE.")
    parser.add_argument("--output-umap",
                        default=os.path.join("outputs", "umap"),
                        help="Directorio de salida UMAP.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _ensure(args.output_tsne)
    _ensure(args.output_umap)

    print("[t-SNE/UMAP] Cargando Breast Cancer Wisconsin...")
    feature_df, labels = load_breast_cancer(args.data)
    X = preprocess(feature_df)
    n_samples, n_features = X.shape
    print(f"[t-SNE/UMAP] Muestras: {n_samples}, Features: {n_features}")

    # ── t-SNE ──
    print("[t-SNE] Ejecutando grid de perplexidades...")
    tsne_results, best_perp = run_tsne_grid(X, labels, args.output_tsne,
                                            args.seed)
    write_tsne_summary(
        os.path.join(args.output_tsne, "tsne_params_silhouette.csv"),
        args.output_tsne, n_samples, n_features)
    print(f"[t-SNE] Mejor perplexity: {best_perp}")

    # ── UMAP ──
    print("[UMAP] Ejecutando grid de parámetros...")
    umap_results, best_umap = run_umap_grid(X, labels, args.output_umap,
                                            args.seed)
    write_umap_summary(
        os.path.join(args.output_umap, "umap_params_silhouette.csv"),
        args.output_umap, n_samples, n_features)
    print(f"[UMAP] Mejor config: n_neighbors={best_umap[0]}, "
          f"min_dist={best_umap[1]}")

    # ── Comparison ──
    print("[Comparación] Generando figura t-SNE vs UMAP...")
    plot_comparison(tsne_results, best_perp, umap_results, best_umap,
                    labels, args.output_umap)

    print(f"[t-SNE] Salidas en: {args.output_tsne}")
    print(f"[UMAP] Salidas en: {args.output_umap}")
    print("[t-SNE/UMAP] Completado.")


if __name__ == "__main__":
    main()
