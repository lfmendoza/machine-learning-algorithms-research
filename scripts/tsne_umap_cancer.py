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
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
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
        ax.set_title(f"perplexity = {perp}  (sil = {sil_val:.3f})", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls="--", lw=0.3)

    fig.suptitle("t-SNE — Efecto de la perplexity", fontsize=13, y=1.01)
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
    ax.set_title(f"t-SNE (perplexity={best_perp}, silhouette={best_sil:.3f})")
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
        ax.set_title(f"n_neighbors={nn}, min_dist=0.1 (sil={sil_val:.3f})",
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
        ax.set_title(f"min_dist={md} (sil={sil_val:.3f})")
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
                 f"sil={best_sil:.3f})")
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

    fig.suptitle("Comparación t-SNE vs UMAP — Breast Cancer Wisconsin",
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
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Resumen t-SNE — Breast Cancer Wisconsin\n\n")
        f.write("## Dataset\n")
        f.write(f"- Fuente: Breast Cancer Wisconsin (Diagnostic)\n")
        f.write(f"- Muestras: {n_samples}\n")
        f.write(f"- Features (tras drop id, diagnosis): {n_features}\n")
        f.write("- Etiquetas: Maligno (M) / Benigno (B)\n\n")
        f.write("## Preprocesamiento\n")
        f.write("- StandardScaler (media=0, std=1) sobre todas las features\n\n")
        f.write("## Parámetros explorados\n")
        f.write(f"- Perplexidades: {metrics['perplexity'].tolist()}\n")
        f.write(f"- Iteraciones: 1000, init: pca, learning_rate: auto\n\n")
        f.write("## Resultados clave\n")
        f.write(f"- Mejor perplexity: {int(best['perplexity'])} "
                f"(silhouette = {best['silhouette']:.3f})\n")
        f.write(f"- KL divergence: {best['kl_divergence']:.4f}\n\n")
        f.write("## Figuras\n")
        f.write("- fig_tsne_01: Comparación de perplexidades\n")
        f.write("- fig_tsne_02: Mejor proyección individual\n")


def write_umap_summary(metrics_path: str, output_dir: str,
                       n_samples: int, n_features: int) -> None:
    metrics = pd.read_csv(metrics_path)
    path = os.path.join(output_dir, "resumen_umap.md")
    best = metrics.sort_values("silhouette", ascending=False).iloc[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Resumen UMAP — Breast Cancer Wisconsin\n\n")
        f.write("## Dataset\n")
        f.write(f"- Fuente: Breast Cancer Wisconsin (Diagnostic)\n")
        f.write(f"- Muestras: {n_samples}\n")
        f.write(f"- Features: {n_features}\n")
        f.write("- Etiquetas: Maligno (M) / Benigno (B)\n\n")
        f.write("## Preprocesamiento\n")
        f.write("- StandardScaler (media=0, std=1)\n\n")
        f.write("## Parámetros explorados\n")
        f.write(f"- n_neighbors: {sorted(metrics['n_neighbors'].unique().tolist())}\n")
        f.write(f"- min_dist: {sorted(metrics['min_dist'].unique().tolist())}\n\n")
        f.write("## Resultados clave\n")
        f.write(f"- Mejor config: n_neighbors={int(best['n_neighbors'])}, "
                f"min_dist={best['min_dist']} "
                f"(silhouette={best['silhouette']:.3f})\n\n")
        f.write("## Figuras\n")
        f.write("- fig_umap_01: Comparación de n_neighbors\n")
        f.write("- fig_umap_02: Efecto de min_dist\n")
        f.write("- fig_umap_03: Mejor proyección individual\n")
        f.write("- fig_comparison_tsne_vs_umap: Comparación lado a lado\n")


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
