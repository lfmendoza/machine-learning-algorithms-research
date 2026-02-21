#!/usr/bin/env python3
"""
Laboratorio de Minería de Datos - Avance
Pipeline reproducible para:
- Carga de dataset (.csv/.xlsx/.sav)
- Limpieza y preprocesamiento (imputación, one-hot, escalado)
- Reducción de dimensionalidad (SVD tipo PCA para matrices dispersas)
- Proyección (t-SNE sobre representación reducida)
- Separación (KMeans + métricas: inertia, silhouette)
- Generación de figuras y tablas para el informe

Uso (ejemplos):
  python scripts/run_avance.py --data data/dataset.csv --sep "," --output outputs
  python scripts/run_avance.py --data data/dataset.sav --output outputs --target "Clase"
  python scripts/run_avance.py --data data/dataset.xlsx --sheet "Hoja1" --output outputs

Requisitos:
  Ver scripts/requirements.txt

Notas:
- Si el dataset es grande, t-SNE puede tardar; use --tsne-sample para muestrear.
- Para reproducibilidad, se fija random_state con --seed.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

try:
    import pyreadstat  # optional; required for .sav
except Exception:
    pyreadstat = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def load_dataset(path: str, sep: str = ",", encoding: str = "utf-8", sheet: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv" or ext == ".txt":
        return pd.read_csv(path, sep=sep, encoding=encoding)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    if ext == ".sav":
        if pyreadstat is None:
            raise RuntimeError("Para leer .sav instale pyreadstat (pip install pyreadstat) y reintente.")
        df, meta = pyreadstat.read_sav(path)
        return df
    raise ValueError(f"Extensión no soportada: {ext}. Use .csv/.xlsx/.sav")


def infer_feature_types(df: pd.DataFrame, target: Optional[str], id_cols: List[str]) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in id_cols and c != target]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def build_preprocess_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


@dataclass
class AvanceConfig:
    data_path: str
    output_dir: str
    sep: str = ","
    encoding: str = "utf-8"
    sheet: Optional[str] = None
    target: Optional[str] = None
    id_cols: List[str] = None
    seed: int = 42
    svd_components: int = 20
    k_min: int = 2
    k_max: int = 10
    tsne_perplexity: float = 30.0
    tsne_iters: int = 1000
    tsne_sample: int = 2000  # rows sampled for t-SNE if dataset is big


def describe_dataset(df: pd.DataFrame, output_dir: str) -> None:
    # Basic shape
    shape_path = os.path.join(output_dir, "dataset_shape.txt")
    with open(shape_path, "w", encoding="utf-8") as f:
        f.write(f"Observaciones (filas): {df.shape[0]}\n")
        f.write(f"Variables (columnas): {df.shape[1]}\n")

    # Dtypes summary
    dtypes_path = os.path.join(output_dir, "variables_y_tipos.csv")
    pd.DataFrame({"variable": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}).to_csv(
        dtypes_path, index=False, encoding="utf-8"
    )

    # Missingness
    missing = df.isna().sum().sort_values(ascending=False)
    missing_path = os.path.join(output_dir, "valores_faltantes.csv")
    missing.to_frame("faltantes").assign(porcentaje=lambda x: (x["faltantes"] / len(df) * 100)).to_csv(
        missing_path, index=True, encoding="utf-8"
    )

    # Missingness plot (top 20)
    top = missing.head(20)
    plt.figure()
    top.iloc[::-1].plot(kind="barh")
    plt.title("Valores faltantes - Top 20 variables")
    plt.xlabel("Cantidad de faltantes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_01_valores_faltantes_top20.png"), dpi=200)
    plt.close()


def correlation_plot(df: pd.DataFrame, output_dir: str, max_features: int = 20) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return
    cols = num_cols[:max_features]
    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(os.path.join(output_dir, "correlacion_numericas.csv"), encoding="utf-8")
    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)
    plt.title(f"Matriz de correlación (primeras {len(cols)} variables numéricas)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_02_correlacion_numericas.png"), dpi=200)
    plt.close()


def run_dimensionality_reduction(X, output_dir: str, n_components: int, seed: int):
    # TruncatedSVD works with sparse/dense matrices
    n_components = max(2, int(n_components))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z = svd.fit_transform(X)

    evr = svd.explained_variance_ratio_
    evr_df = pd.DataFrame({
        "componente": [f"C{i+1}" for i in range(len(evr))],
        "varianza_explicada": evr,
        "varianza_explicada_acumulada": np.cumsum(evr),
    })
    evr_df.to_csv(os.path.join(output_dir, "svd_varianza_explicada.csv"), index=False, encoding="utf-8")

    # Plot explained variance cumulative
    plt.figure()
    plt.plot(range(1, len(evr) + 1), np.cumsum(evr), marker="o")
    plt.title("Varianza explicada acumulada (SVD tipo PCA)")
    plt.xlabel("Número de componentes")
    plt.ylabel("Varianza explicada acumulada")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_03_varianza_explicada_acumulada.png"), dpi=200)
    plt.close()

    return svd, Z


def run_kmeans(Z: np.ndarray, output_dir: str, k_min: int, k_max: int, seed: int):
    ks = list(range(int(k_min), int(k_max) + 1))
    inertia = []
    sil = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(Z)
        inertia.append(km.inertia_)
        # silhouette requires at least 2 clusters and less than n_samples
        if len(np.unique(labels)) > 1 and Z.shape[0] > k:
            sil.append(silhouette_score(Z, labels))
        else:
            sil.append(np.nan)

    metrics = pd.DataFrame({"k": ks, "inertia": inertia, "silhouette": sil})
    metrics.to_csv(os.path.join(output_dir, "kmeans_metricas.csv"), index=False, encoding="utf-8")

    # Elbow plot
    plt.figure()
    plt.plot(ks, inertia, marker="o")
    plt.title("KMeans - Inercia vs k (Elbow)")
    plt.xlabel("k")
    plt.ylabel("Inercia")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_04_kmeans_elbow.png"), dpi=200)
    plt.close()

    # Silhouette plot
    plt.figure()
    plt.plot(ks, sil, marker="o")
    plt.title("KMeans - Silhouette vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_05_kmeans_silhouette.png"), dpi=200)
    plt.close()

    # Select best k by silhouette (ignore NaN)
    best_row = metrics.dropna().sort_values("silhouette", ascending=False).head(1)
    if len(best_row) == 0:
        best_k = ks[0]
    else:
        best_k = int(best_row["k"].iloc[0])

    best_km = KMeans(n_clusters=best_k, n_init=20, random_state=seed)
    labels = best_km.fit_predict(Z)

    with open(os.path.join(output_dir, "kmeans_mejor_k.txt"), "w", encoding="utf-8") as f:
        f.write(f"Mejor k (por silhouette): {best_k}\n")
        if len(best_row) > 0:
            f.write(f"Silhouette: {best_row['silhouette'].iloc[0]:.4f}\n")

    return labels, best_k


def plot_2d_scatter(Z: np.ndarray, labels: Optional[np.ndarray], output_path: str, title: str) -> None:
    plt.figure()
    if labels is None:
        plt.scatter(Z[:, 0], Z[:, 1], s=10)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=10)
    plt.title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_tsne(Z: np.ndarray, output_dir: str, seed: int, perplexity: float, n_iter: int, sample: int):
    n = Z.shape[0]
    if n > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample, replace=False)
        Zs = Z[idx]
        idx_path = os.path.join(output_dir, "tsne_muestra_indices.csv")
        pd.DataFrame({"row_index": idx}).to_csv(idx_path, index=False, encoding="utf-8")
    else:
        Zs = Z
        idx = None

    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        n_iter=int(n_iter),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    T = tsne.fit_transform(Zs)
    pd.DataFrame({"tsne_1": T[:, 0], "tsne_2": T[:, 1]}).to_csv(os.path.join(output_dir, "tsne_2d.csv"), index=False, encoding="utf-8")

    plt.figure()
    plt.scatter(T[:, 0], T[:, 1], s=10)
    plt.title("Proyección t-SNE (2D) sobre representación reducida")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_06_tsne_2d.png"), dpi=200)
    plt.close()

    return idx


def cluster_profiles(df_original: pd.DataFrame, labels: np.ndarray, output_dir: str, max_categories: int = 3) -> None:
    dfp = df_original.copy()
    dfp["_cluster"] = labels

    num_cols = [c for c in df_original.columns if pd.api.types.is_numeric_dtype(df_original[c])]
    if len(num_cols) > 0:
        means = dfp.groupby("_cluster")[num_cols].mean(numeric_only=True)
        means.to_csv(os.path.join(output_dir, "clusters_promedios_numericas.csv"), encoding="utf-8")

    # Top categories per cluster for categorical columns (if any)
    cat_cols = [c for c in df_original.columns if c not in num_cols]
    rows = []
    for c in cat_cols:
        # skip very high cardinality columns
        if df_original[c].nunique(dropna=True) > 200:
            continue
        for cl in sorted(dfp["_cluster"].unique()):
            vc = dfp.loc[dfp["_cluster"] == cl, c].value_counts(dropna=True).head(max_categories)
            for k, v in vc.items():
                rows.append({"cluster": cl, "variable": c, "categoria": str(k), "frecuencia": int(v)})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "clusters_top_categorias.csv"), index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de avance para laboratorio de Minería de Datos.")
    parser.add_argument("--data", required=True, help="Ruta al dataset (.csv/.xlsx/.sav).")
    parser.add_argument("--output", default="outputs", help="Directorio de salida para tablas/figuras.")
    parser.add_argument("--sep", default=",", help="Separador para CSV (por defecto ,).")
    parser.add_argument("--encoding", default="utf-8", help="Encoding para CSV (por defecto utf-8).")
    parser.add_argument("--sheet", default=None, help="Nombre de hoja (si .xlsx).")
    parser.add_argument("--target", default=None, help="Nombre de columna objetivo/etiqueta (opcional).")
    parser.add_argument("--id-cols", default="", help="Lista separada por coma de columnas ID a excluir (opcional).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--svd-components", type=int, default=20, help="Número de componentes SVD (>=2).")
    parser.add_argument("--k-min", type=int, default=2, help="k mínimo para KMeans.")
    parser.add_argument("--k-max", type=int, default=10, help="k máximo para KMeans.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="Perplexity t-SNE.")
    parser.add_argument("--tsne-iters", type=int, default=1000, help="Iteraciones t-SNE.")
    parser.add_argument("--tsne-sample", type=int, default=2000, help="Muestra máxima de filas para t-SNE.")
    args = parser.parse_args()

    cfg = AvanceConfig(
        data_path=args.data,
        output_dir=args.output,
        sep=args.sep,
        encoding=args.encoding,
        sheet=args.sheet,
        target=args.target,
        id_cols=[c.strip() for c in args.id_cols.split(",") if c.strip()],
        seed=args.seed,
        svd_components=args.svd_components,
        k_min=args.k_min,
        k_max=args.k_max,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iters=args.tsne_iters,
        tsne_sample=args.tsne_sample,
    )

    _ensure_dir(cfg.output_dir)

    df = load_dataset(cfg.data_path, sep=cfg.sep, encoding=cfg.encoding, sheet=cfg.sheet)

    # Store raw snapshot (first rows)
    df.head(20).to_csv(os.path.join(cfg.output_dir, "muestra_20_filas.csv"), index=False, encoding="utf-8")

    describe_dataset(df, cfg.output_dir)
    correlation_plot(df, cfg.output_dir)

    num_cols, cat_cols = infer_feature_types(df, cfg.target, cfg.id_cols or [])

    # Separate y if provided (only for coloring/interpretation, not supervised learning)
    y = None
    if cfg.target and cfg.target in df.columns:
        y = df[cfg.target].copy()

    X_df = df.drop(columns=[c for c in (cfg.id_cols or []) if c in df.columns], errors="ignore")
    if cfg.target and cfg.target in X_df.columns:
        X_df = X_df.drop(columns=[cfg.target], errors="ignore")

    preprocess = build_preprocess_pipeline(num_cols, cat_cols)
    X = preprocess.fit_transform(X_df)

    svd, Z = run_dimensionality_reduction(X, cfg.output_dir, cfg.svd_components, cfg.seed)

    # KMeans on reduced space
    labels, best_k = run_kmeans(Z, cfg.output_dir, cfg.k_min, cfg.k_max, cfg.seed)

    # Scatter in component space (colored by cluster)
    plot_2d_scatter(Z, labels, os.path.join(cfg.output_dir, "fig_07_componentes_1_2_por_cluster.png"),
                    "Proyección en Componentes 1-2 (coloreado por cluster KMeans)")

    # Scatter colored by target if available
    if y is not None:
        # map categories to integers for coloring
        y_codes, uniques = pd.factorize(y.astype(str), sort=True)
        pd.DataFrame({"categoria": uniques, "codigo": range(len(uniques))}).to_csv(
            os.path.join(cfg.output_dir, "target_codigos.csv"), index=False, encoding="utf-8"
        )
        plot_2d_scatter(Z, y_codes, os.path.join(cfg.output_dir, "fig_08_componentes_1_2_por_target.png"),
                        f"Proyección en Componentes 1-2 (coloreado por {cfg.target})")

    # t-SNE projection
    run_tsne(Z, cfg.output_dir, cfg.seed, cfg.tsne_perplexity, cfg.tsne_iters, cfg.tsne_sample)

    # Cluster profiles on original df (excluding target/id? keep original for interpretation)
    cluster_profiles(df.drop(columns=[cfg.target], errors="ignore") if cfg.target else df, labels, cfg.output_dir)

    # Summary for report
    summary_path = os.path.join(cfg.output_dir, "resumen_para_informe.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Resumen para el informe (copiar/pegar)\n\n")
        f.write(f"- Observaciones: {df.shape[0]}\n")
        f.write(f"- Variables: {df.shape[1]}\n")
        f.write(f"- Variables numéricas: {len(num_cols)}\n")
        f.write(f"- Variables categóricas: {len(cat_cols)}\n")
        f.write(f"- SVD componentes: {cfg.svd_components}\n")
        f.write(f"- KMeans mejor k (por silhouette): {best_k}\n")
        f.write("\nArchivos clave generados:\n")
        for name in [
            "fig_01_valores_faltantes_top20.png",
            "fig_03_varianza_explicada_acumulada.png",
            "fig_04_kmeans_elbow.png",
            "fig_05_kmeans_silhouette.png",
            "fig_06_tsne_2d.png",
            "fig_07_componentes_1_2_por_cluster.png",
        ]:
            f.write(f"- {name}\n")

    print("OK. Salidas generadas en:", cfg.output_dir)


if __name__ == "__main__":
    main()
