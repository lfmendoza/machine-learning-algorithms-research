#!/usr/bin/env python3
"""
SVD (Singular Value Decomposition) aplicado al dataset MovieLens 100k.

Construye la matriz dispersa usuario-película, aplica TruncatedSVD y genera
figuras y tablas para el informe del laboratorio.

Uso:
  python scripts/svd_movielens.py
  python scripts/svd_movielens.py --components 50 --seed 42
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings(data_dir: str) -> pd.DataFrame:
    """Carga el archivo u.data (tab-separated: user_id, item_id, rating, timestamp)."""
    path = os.path.join(data_dir, "u.data")
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def load_items(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "u.item")
    columns = ["item_id", "title", "release_date", "video_release_date",
                "url"] + GENRE_NAMES
    df = pd.read_csv(
        path, sep="|", header=None, names=columns,
        encoding="latin-1",
    )
    return df


def build_user_item_matrix(ratings: pd.DataFrame):
    """Construye la matriz dispersa usuario-película en formato CSR.

    Los IDs de usuario/película se indexan desde 0. Las celdas vacías
    (películas no calificadas) quedan como ceros en la matriz dispersa.
    """
    users = ratings["user_id"].values - 1
    items = ratings["item_id"].values - 1
    vals = ratings["rating"].values.astype(np.float64)

    n_users = ratings["user_id"].max()
    n_items = ratings["item_id"].max()

    # CSR (Compressed Sparse Row) almacena eficientemente matrices con
    # muchos ceros — ideal para matrices de ratings (~94% vacía)
    R = csr_matrix((vals, (users, items)), shape=(n_users, n_items))
    return R


def run_svd(R, n_components: int, seed: int):
    """Aplica TruncatedSVD a la matriz dispersa.

    TruncatedSVD no centra los datos (no resta la media), lo que lo hace
    adecuado para matrices dispersas donde el centrado destruiría la
    dispersión. Retorna la proyección de usuarios U_reduced = U·Σ.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    U_reduced = svd.fit_transform(R)
    return svd, U_reduced


def plot_explained_variance(svd, output_dir: str) -> pd.DataFrame:
    evr = svd.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    df = pd.DataFrame({
        "componente": np.arange(1, len(evr) + 1),
        "varianza_explicada": evr,
        "varianza_acumulada": cumulative,
    })
    df.to_csv(os.path.join(output_dir, "svd_varianza_explicada.csv"),
              index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(df["componente"], df["varianza_explicada"], color="steelblue")
    axes[0].set_title("Varianza explicada por componente")
    axes[0].set_xlabel("Componente")
    axes[0].set_ylabel("Proporción de varianza")

    axes[1].plot(df["componente"], df["varianza_acumulada"],
                 marker="o", markersize=3, color="darkorange")
    axes[1].set_title("Varianza explicada acumulada")
    axes[1].set_xlabel("Número de componentes")
    axes[1].set_ylabel("Varianza acumulada")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.80, ls="--", color="gray", lw=0.8, label="80 %")
    axes[1].axhline(0.90, ls="--", color="silver", lw=0.8, label="90 %")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", lw=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_svd_01_varianza_explicada.png"),
                dpi=200)
    plt.close(fig)
    return df


def plot_users_2d(U_reduced, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(U_reduced[:, 0], U_reduced[:, 1], s=4, alpha=0.4,
               color="steelblue")
    ax.set_title("Usuarios proyectados en las primeras 2 componentes SVD")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(True, ls="--", lw=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_svd_02_usuarios_2d.png"),
                dpi=200)
    plt.close(fig)


def plot_items_2d(svd, items_df: pd.DataFrame, output_dir: str) -> None:
    """Proyecta películas en el espacio latente V^T y colorea por género principal."""
    Vt = svd.components_  # (n_components, n_items)
    item_coords = Vt[:2, :].T  # (n_items, 2)

    genre_cols = GENRE_NAMES
    genre_matrix = items_df[genre_cols].values
    primary_genre = np.argmax(genre_matrix, axis=1)

    n_genres = len(genre_cols)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_genres)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(item_coords[:, 0], item_coords[:, 1],
                         c=primary_genre, cmap=cmap, s=8, alpha=0.6,
                         vmin=0, vmax=n_genres - 1)

    top_genres = pd.Series(primary_genre).value_counts().head(8).index.tolist()
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(g),
                          markersize=5, label=genre_cols[g])
               for g in sorted(top_genres)]
    ax.legend(handles=handles, fontsize=7, loc="best", title="Género principal",
              title_fontsize=8, ncol=2)

    ax.set_title("Películas proyectadas en las primeras 2 componentes SVD")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(True, ls="--", lw=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_svd_03_peliculas_2d.png"),
                dpi=200)
    plt.close(fig)


def plot_reconstruction_error(R, output_dir: str, max_k: int, seed: int) -> None:
    """Calcula error de reconstrucción (Frobenius) para distintos k."""
    from scipy.sparse import issparse

    R_dense = R.toarray() if issparse(R) else R
    frobenius_total = np.linalg.norm(R_dense, "fro")

    ks = list(range(2, max_k + 1, max(1, max_k // 15)))
    if ks[-1] != max_k:
        ks.append(max_k)

    errors = []
    for k in ks:
        svd_k = TruncatedSVD(n_components=k, random_state=seed)
        Z = svd_k.fit_transform(R)
        R_approx = Z @ svd_k.components_
        err = np.linalg.norm(R_dense - R_approx, "fro") / frobenius_total
        errors.append(err)

    df = pd.DataFrame({"k": ks, "error_relativo": errors})
    df.to_csv(os.path.join(output_dir, "svd_reconstruccion_error.csv"),
              index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, errors, marker="o", markersize=4, color="crimson")
    ax.set_title("Error relativo de reconstrucción vs. componentes SVD")
    ax.set_xlabel("Número de componentes (k)")
    ax.set_ylabel("||R - R_k||_F / ||R||_F")
    ax.grid(True, ls="--", lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_svd_04_reconstruccion_error.png"),
                dpi=200)
    plt.close(fig)


def top_movies_per_component(svd, items_df: pd.DataFrame,
                             output_dir: str, top_n: int = 10) -> None:
    """Para cada componente, lista las películas con mayor peso absoluto."""
    Vt = svd.components_
    rows = []
    for comp_idx in range(min(Vt.shape[0], 10)):
        weights = Vt[comp_idx]
        top_idx = np.argsort(np.abs(weights))[::-1][:top_n]
        for rank, idx in enumerate(top_idx, 1):
            if idx < len(items_df):
                rows.append({
                    "componente": comp_idx + 1,
                    "rank": rank,
                    "item_id": int(items_df.iloc[idx]["item_id"]),
                    "titulo": items_df.iloc[idx]["title"],
                    "peso": float(weights[idx]),
                })

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "svd_top_peliculas_por_componente.csv"),
        index=False, encoding="utf-8")


def write_summary(evr_df: pd.DataFrame, n_components: int,
                  n_users: int, n_items: int, n_ratings: int,
                  output_dir: str) -> None:
    path = os.path.join(output_dir, "resumen_svd.md")
    rows80 = evr_df.loc[evr_df["varianza_acumulada"] >= 0.80, "componente"]
    var80 = int(rows80.min()) if len(rows80) > 0 else None
    rows90 = evr_df.loc[evr_df["varianza_acumulada"] >= 0.90, "componente"]
    var90 = int(rows90.min()) if len(rows90) > 0 else None
    top1_var = evr_df.iloc[0]["varianza_explicada"] * 100
    top5_var = evr_df.iloc[:5]["varianza_explicada"].sum() * 100
    total_var = evr_df.iloc[-1]["varianza_acumulada"] * 100

    with open(path, "w", encoding="utf-8") as f:
        f.write("# SVD (Descomposición en Valores Singulares)\n\n")

        # ── 1. Descripción teórica ──
        f.write("## 1. Descripción teórica\n\n")
        f.write("### Explicación del algoritmo y objetivo principal\n\n")
        f.write(
            "La Descomposición en Valores Singulares (SVD) factoriza una "
            "matriz A de dimensiones m×n en tres matrices: A = U·Σ·Vᵀ, donde "
            "U (m×m) contiene los vectores singulares izquierdos, Σ (m×n) es "
            "una matriz diagonal con los valores singulares ordenados de mayor "
            "a menor, y Vᵀ (n×n) contiene los vectores singulares derechos. "
            "Su objetivo principal es la reducción de dimensionalidad: al "
            "retener solo los k valores singulares más grandes se obtiene la "
            "mejor aproximación de rango k en norma de Frobenius.\n\n")
        f.write("### Principales características y supuestos\n\n")
        f.write(
            "- Es un método determinístico y algebraicamente exacto (no "
            "iterativo en su forma teórica).\n"
            "- No requiere que los datos sigan una distribución particular "
            "(no asume normalidad).\n"
            "- Opera directamente sobre la matriz de datos, sin necesidad de "
            "calcular la matriz de covarianza.\n"
            "- La variante TruncatedSVD utilizada aquí trabaja eficientemente "
            "con matrices dispersas, ya que no centra los datos (no resta la "
            "media), a diferencia de PCA.\n"
            "- Los valores singulares reflejan la importancia relativa de cada "
            "componente: σ₁ ≥ σ₂ ≥ ... ≥ σₖ.\n\n")
        f.write("### Diferencias con PCA\n\n")
        f.write(
            "| Aspecto | SVD (TruncatedSVD) | PCA |\n"
            "|---|---|---|\n"
            "| Centrado | No centra los datos | Centra (resta la media) |\n"
            "| Matrices dispersas | Soporte nativo (no densifica) | Requiere "
            "densificar o usar variantes |\n"
            "| Base matemática | Factorización directa A = UΣVᵀ | "
            "Diagonalización de la covarianza Cov = VΛVᵀ |\n"
            "| Interpretación | Factores latentes de la matriz original | "
            "Direcciones de máxima varianza centrada |\n"
            "| Caso especial | PCA es SVD aplicado a datos centrados | — |\n\n")

        # ── 2. Usos y aplicaciones ──
        f.write("## 2. Usos y aplicaciones\n\n")
        f.write("### Principales usos en análisis de datos\n\n")
        f.write(
            "- **Reducción de dimensionalidad**: comprimir datos de alta "
            "dimensión preservando la mayor varianza posible.\n"
            "- **Sistemas de recomendación**: factorización de matrices "
            "usuario-ítem para descubrir factores latentes (gustos, "
            "categorías implícitas).\n"
            "- **Compresión de datos e imágenes**: aproximaciones de bajo "
            "rango para almacenamiento eficiente.\n"
            "- **Procesamiento de lenguaje natural (LSA/LSI)**: reducir la "
            "matriz término-documento para capturar relaciones semánticas.\n\n")
        f.write("### Áreas de aplicación\n\n")
        f.write(
            "1. **Sistemas de recomendación (Netflix, Spotify)**: SVD "
            "identifica factores latentes en matrices de ratings para predecir "
            "preferencias no observadas. Es la base del filtrado colaborativo "
            "matricial.\n"
            "2. **Procesamiento de imágenes y visión por computadora**: la "
            "aproximación de bajo rango permite comprimir imágenes reteniendo "
            "las estructuras visuales más relevantes, y se usa en "
            "reconocimiento facial (eigenfaces).\n"
            "3. **Bioinformática**: análisis de matrices de expresión génica "
            "para identificar patrones de co-expresión entre genes y "
            "condiciones experimentales.\n\n")

        # ── 3. Aplicación práctica ──
        f.write("## 3. Aplicación práctica\n\n")
        f.write("### Dataset utilizado\n\n")
        f.write(
            f"- **Fuente**: MovieLens 100k (GroupLens Research, University of "
            f"Minnesota)\n"
            f"- **Usuarios**: {n_users}\n"
            f"- **Películas**: {n_items}\n"
            f"- **Ratings totales**: {n_ratings:,}\n"
            f"- **Escala de ratings**: 1 a 5 (enteros)\n"
            f"- **Densidad de la matriz**: "
            f"{n_ratings / (n_users * n_items) * 100:.2f}% "
            f"(altamente dispersa)\n\n")
        f.write("### Decisiones de preprocesamiento\n\n")
        f.write(
            "- Se construyó una matriz dispersa usuario-película en formato "
            "CSR (Compressed Sparse Row) de {n_users}×{n_items}.\n"
            "- Se utilizaron los ratings directos como valores (sin centrar), "
            "apropiado para TruncatedSVD sobre matrices dispersas.\n"
            "- Se solicitaron {n_comp} componentes para el análisis.\n\n"
            .format(n_users=n_users, n_items=n_items, n_comp=n_components))
        f.write("### Resultados obtenidos\n\n")
        f.write(
            f"- **Componentes utilizados**: {n_components}\n"
            f"- **Varianza explicada por el 1er componente**: {top1_var:.2f}%\n"
            f"- **Varianza acumulada (primeros 5 componentes)**: "
            f"{top5_var:.2f}%\n"
            f"- **Varianza acumulada total ({n_components} componentes)**: "
            f"{total_var:.2f}%\n")
        var80_str = (str(var80) if var80 is not None
                     else f">{n_components} (no alcanzado con {n_components} "
                          f"componentes)")
        var90_str = (str(var90) if var90 is not None
                     else f">{n_components} (no alcanzado con {n_components} "
                          f"componentes)")
        f.write(
            f"- **Componentes necesarios para 80% de varianza**: "
            f"{var80_str}\n"
            f"- **Componentes necesarios para 90% de varianza**: "
            f"{var90_str}\n\n")
        f.write("### Interpretación\n\n")
        f.write(
            "Los primeros componentes capturan los patrones de rating más "
            "globales (e.g., películas populares universalmente bien "
            "calificadas), mientras que los componentes posteriores capturan "
            "preferencias más específicas de nichos o géneros. La proyección "
            "2D de películas (fig_svd_03) muestra agrupamientos por género, "
            "lo que confirma que SVD descubre factores latentes con "
            "interpretación semántica. El error de reconstrucción "
            "(fig_svd_04) decrece rápidamente con los primeros componentes, "
            "indicando que la información esencial de la matriz se concentra "
            "en pocas dimensiones. La tabla de top películas por componente "
            "revela qué títulos dominan cada factor latente.\n\n"
            f"La varianza acumulada con {n_components} componentes alcanza "
            f"solo el {total_var:.2f}%, lo cual es esperado dado que la "
            f"matriz usuario-película es altamente dispersa (densidad "
            f"~{n_ratings / (n_users * n_items) * 100:.1f}%) y contiene "
            f"mucha variabilidad individual. Esto implica que se necesitarían "
            f"muchos más componentes para capturar la mayoría de la varianza, "
            f"pero los primeros componentes ya contienen los patrones más "
            f"informativos para recomendación.\n\n")
        f.write("### Limitaciones\n\n")
        f.write(
            "- SVD asume una relación lineal entre los factores latentes; no "
            "captura interacciones no lineales en las preferencias de los "
            "usuarios.\n"
            "- La matriz de ratings tiene valores faltantes (celdas vacías = "
            "no calificado), que TruncatedSVD trata como ceros; esto puede "
            "sesgar los factores hacia películas populares con más ratings.\n"
            "- No considera información temporal: las preferencias de los "
            "usuarios pueden cambiar con el tiempo.\n"
            "- La interpretación de los factores latentes es subjetiva; no "
            "siempre corresponden a conceptos claros como géneros.\n"
            "- Con matrices muy dispersas como esta (~6% de densidad), la "
            "varianza explicada crece lentamente con el número de "
            "componentes.\n\n")
        f.write("### Figuras generadas\n\n")
        f.write(
            "| Figura | Descripción |\n"
            "|---|---|\n"
            "| fig_svd_01 | Varianza explicada por componente y acumulada |\n"
            "| fig_svd_02 | Usuarios proyectados en espacio latente 2D |\n"
            "| fig_svd_03 | Películas proyectadas en 2D, coloreadas por "
            "género |\n"
            "| fig_svd_04 | Error relativo de reconstrucción vs. componentes "
            "|\n\n")
        f.write("### Tablas generadas\n\n")
        f.write(
            "| Tabla | Contenido |\n"
            "|---|---|\n"
            "| svd_varianza_explicada.csv | Varianza explicada y acumulada "
            "por componente |\n"
            "| svd_reconstruccion_error.csv | Error de reconstrucción para "
            "distintos k |\n"
            "| svd_top_peliculas_por_componente.csv | Top 10 películas con "
            "mayor peso por factor latente |\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVD sobre MovieLens 100k.")
    parser.add_argument("--data-dir",
                        default=os.path.join("datasets", "movie-lens"),
                        help="Directorio con archivos MovieLens.")
    parser.add_argument("--output",
                        default=os.path.join("outputs", "svd"),
                        help="Directorio de salida.")
    parser.add_argument("--components", type=int, default=50,
                        help="Número de componentes SVD.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _ensure(args.output)

    print("[SVD] Cargando datos MovieLens...")
    ratings = load_ratings(args.data_dir)
    items = load_items(args.data_dir)

    print(f"[SVD] Ratings: {len(ratings)}, Usuarios: {ratings['user_id'].nunique()}, "
          f"Películas: {ratings['item_id'].nunique()}")

    R = build_user_item_matrix(ratings)
    print(f"[SVD] Matriz usuario-película: {R.shape}, "
          f"densidad: {R.nnz / (R.shape[0] * R.shape[1]):.4f}")

    n_comp = min(args.components, min(R.shape) - 1)
    print(f"[SVD] Aplicando TruncatedSVD con {n_comp} componentes...")
    svd, U_reduced = run_svd(R, n_comp, args.seed)

    print("[SVD] Generando figuras y tablas...")
    evr_df = plot_explained_variance(svd, args.output)
    plot_users_2d(U_reduced, args.output)
    plot_items_2d(svd, items, args.output)
    plot_reconstruction_error(R, args.output, max_k=n_comp, seed=args.seed)
    top_movies_per_component(svd, items, args.output)
    write_summary(evr_df, n_comp,
                  ratings["user_id"].nunique(),
                  ratings["item_id"].nunique(),
                  len(ratings), args.output)

    print(f"[SVD] Completado. Salidas en: {args.output}")


if __name__ == "__main__":
    main()
