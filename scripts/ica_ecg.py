#!/usr/bin/env python3
"""
ICA (Independent Component Analysis) aplicado a señales ECG del
MIT-BIH Arrhythmia Database con anotaciones P-Wave de PhysioNet.

Demuestra la separación de fuentes independientes (componentes cardíacas)
a partir de las derivaciones ECG multicanal, y superpone las anotaciones
de onda P como validación.

Uso:
  python scripts/ica_ecg.py
  python scripts/ica_ecg.py --data-dir datasets/physionet-ecg --seed 42
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

import wfdb


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


RECORDS = ["100", "101", "103", "106", "117", "119",
           "122", "207", "214", "222", "223", "231"]

# Registros seleccionados para las figuras detalladas (variedad de patologías)
DEMO_RECORDS = ["100", "119", "207"]

SAMPLE_WINDOW = 3600  # 10 segundos a 360 Hz


# ─────────────────────────── Data loading ───────────────────────────

def load_record(data_dir: str, rec: str, start: int = 0,
                length: int | None = None):
    """Lee señales ECG y devuelve (signal_array, record_obj)."""
    rec_path = os.path.join(data_dir, rec)
    if length is None:
        record = wfdb.rdrecord(rec_path)
    else:
        record = wfdb.rdrecord(rec_path, sampfrom=start,
                               sampto=start + length)
    return record.p_signal, record


def load_pwave_annotations(data_dir: str, rec: str):
    """Intenta cargar anotaciones P-wave; devuelve None si no existen."""
    rec_path = os.path.join(data_dir, rec)
    try:
        ann = wfdb.rdann(rec_path, "pwave")
        return ann
    except Exception:
        return None


# ─────────────────────────── ICA core ───────────────────────────────

def apply_ica(signals: np.ndarray, n_components: int, seed: int):
    """Aplica FastICA y devuelve (componentes, modelo ICA, scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(signals)

    ica = FastICA(n_components=n_components, random_state=seed,
                  max_iter=1000, whiten="unit-variance")
    S = ica.fit_transform(X)
    return S, ica, scaler


# ─────────────────────── Plotting helpers ───────────────────────────

def plot_original_signals(signals: np.ndarray, sig_names: list[str],
                          fs: int, rec_name: str, output_dir: str,
                          pwave_ann=None) -> None:
    """Grafica las señales ECG originales (multicanal)."""
    n_ch = signals.shape[1]
    t = np.arange(signals.shape[0]) / fs

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, signals[:, i], lw=0.5, color="steelblue")
        ax.set_ylabel(sig_names[i], fontsize=9)
        ax.grid(True, ls="--", lw=0.3)

        if pwave_ann is not None:
            mask = (pwave_ann.sample >= 0) & (pwave_ann.sample < signals.shape[0])
            pw_times = pwave_ann.sample[mask] / fs
            for pt in pw_times:
                ax.axvline(pt, color="red", alpha=0.3, lw=0.4)

    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"Señales ECG originales — Registro {rec_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"fig_ica_01_originales_{rec_name}.png"), dpi=200)
    plt.close(fig)


def plot_ica_components(S: np.ndarray, fs: int, rec_name: str,
                        output_dir: str, pwave_ann=None) -> None:
    """Grafica componentes independientes."""
    n_comp = S.shape[1]
    t = np.arange(S.shape[0]) / fs

    fig, axes = plt.subplots(n_comp, 1, figsize=(12, 2.5 * n_comp),
                             sharex=True)
    if n_comp == 1:
        axes = [axes]

    colors = ["#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]
    for i, ax in enumerate(axes):
        ax.plot(t, S[:, i], lw=0.5, color=colors[i % len(colors)])
        ax.set_ylabel(f"IC {i + 1}", fontsize=9)
        ax.grid(True, ls="--", lw=0.3)

        if pwave_ann is not None:
            mask = (pwave_ann.sample >= 0) & (pwave_ann.sample < S.shape[0])
            pw_times = pwave_ann.sample[mask] / fs
            for pt in pw_times:
                ax.axvline(pt, color="red", alpha=0.3, lw=0.4)

    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"Componentes independientes (FastICA) — Registro {rec_name}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"fig_ica_02_componentes_{rec_name}.png"), dpi=200)
    plt.close(fig)


def plot_comparison(signals: np.ndarray, S: np.ndarray, sig_names: list[str],
                    fs: int, rec_name: str, output_dir: str) -> None:
    """Comparación lado a lado: originales vs ICA."""
    n_ch = signals.shape[1]
    t = np.arange(signals.shape[0]) / fs

    fig, axes = plt.subplots(n_ch, 2, figsize=(14, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_ch):
        axes[i, 0].plot(t, signals[:, i], lw=0.5, color="steelblue")
        axes[i, 0].set_ylabel(sig_names[i], fontsize=9)
        axes[i, 0].grid(True, ls="--", lw=0.3)
        if i == 0:
            axes[i, 0].set_title("Señales originales", fontsize=10)

        axes[i, 1].plot(t, S[:, i], lw=0.5, color="#2ca02c")
        axes[i, 1].set_ylabel(f"IC {i + 1}", fontsize=9)
        axes[i, 1].grid(True, ls="--", lw=0.3)
        if i == 0:
            axes[i, 1].set_title("Componentes independientes", fontsize=10)

    axes[-1, 0].set_xlabel("Tiempo (s)")
    axes[-1, 1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"Original vs ICA — Registro {rec_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"fig_ica_03_comparacion_{rec_name}.png"), dpi=200)
    plt.close(fig)


def plot_pwave_detail(signals: np.ndarray, S: np.ndarray,
                      sig_names: list[str], fs: int, rec_name: str,
                      pwave_ann, output_dir: str) -> None:
    """Zoom sobre una ventana con ondas P anotadas."""
    if pwave_ann is None or len(pwave_ann.sample) == 0:
        return

    valid_pw = pwave_ann.sample[
        (pwave_ann.sample >= 0) & (pwave_ann.sample < signals.shape[0])]
    if len(valid_pw) < 3:
        return

    # Centrar en la tercera onda P, ventana de ~3 segundos
    center = valid_pw[2]
    half_win = int(1.5 * fs)
    start = max(0, center - half_win)
    end = min(signals.shape[0], center + half_win)
    t = np.arange(start, end) / fs

    n_ch = signals.shape[1]
    fig, axes = plt.subplots(n_ch, 2, figsize=(14, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_ch):
        axes[i, 0].plot(t, signals[start:end, i], lw=0.6, color="steelblue")
        axes[i, 0].set_ylabel(sig_names[i], fontsize=9)
        axes[i, 0].grid(True, ls="--", lw=0.3)

        axes[i, 1].plot(t, S[start:end, i], lw=0.6, color="#2ca02c")
        axes[i, 1].set_ylabel(f"IC {i+1}", fontsize=9)
        axes[i, 1].grid(True, ls="--", lw=0.3)

    # P-wave markers
    pw_in_window = valid_pw[(valid_pw >= start) & (valid_pw < end)]
    for pw in pw_in_window:
        pw_t = pw / fs
        for i in range(n_ch):
            axes[i, 0].axvline(pw_t, color="red", alpha=0.6, lw=1,
                               label="P-wave" if i == 0 else None)
            axes[i, 1].axvline(pw_t, color="red", alpha=0.6, lw=1)

    axes[0, 0].set_title("Señales originales", fontsize=10)
    axes[0, 1].set_title("Componentes independientes", fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("Tiempo (s)")
    axes[-1, 1].set_xlabel("Tiempo (s)")

    fig.suptitle(f"Detalle P-wave — Registro {rec_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"fig_ica_05_pwave_{rec_name}.png"), dpi=200)
    plt.close(fig)


# ─────────────────────── Kurtosis analysis ──────────────────────────

def compute_kurtosis_all(all_ica_results: dict, output_dir: str) -> pd.DataFrame:
    """Calcula kurtosis (medida de no-gaussianidad) de cada componente."""
    rows = []
    for rec_name, (signals, S, sig_names) in all_ica_results.items():
        for i in range(S.shape[1]):
            k_ic = float(sp_kurtosis(S[:, i], fisher=True))
            k_orig = float(sp_kurtosis(signals[:, i], fisher=True))
            rows.append({
                "registro": rec_name,
                "canal_original": sig_names[i],
                "kurtosis_original": round(k_orig, 4),
                "componente_ic": f"IC {i+1}",
                "kurtosis_ic": round(k_ic, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "ica_kurtosis.csv"),
              index=False, encoding="utf-8")
    return df


def plot_kurtosis(kurtosis_df: pd.DataFrame, output_dir: str) -> None:
    records = kurtosis_df["registro"].unique()
    n_rec = len(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_rec)
    width = 0.2

    for i, col_type in enumerate(["kurtosis_original", "kurtosis_ic"]):
        ic_idx = 0 if "original" in col_type else 0
        for ch in range(2):
            subset = kurtosis_df.groupby("registro").nth(ch)
            if col_type not in subset.columns:
                continue
            vals = [subset.loc[r, col_type] if r in subset.index else 0
                    for r in records]
            label_prefix = "Original" if "original" in col_type else "IC"
            offset = i * 2 * width + ch * width
            ax.bar(x + offset - 1.5 * width, vals, width * 0.9,
                   label=f"{label_prefix} ch{ch + 1}")

    ax.set_xticks(x)
    ax.set_xticklabels(records, fontsize=8)
    ax.set_xlabel("Registro")
    ax.set_ylabel("Kurtosis (Fisher)")
    ax.set_title("Kurtosis: canales originales vs componentes ICA")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, ls="--", lw=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_ica_04_kurtosis.png"), dpi=200)
    plt.close(fig)


def save_mixing_matrix(ica_model, rec_name: str, output_dir: str) -> None:
    A = ica_model.mixing_
    df = pd.DataFrame(A, columns=[f"IC_{i+1}" for i in range(A.shape[1])],
                      index=[f"Canal_{i+1}" for i in range(A.shape[0])])
    df.to_csv(os.path.join(output_dir, f"ica_mixing_matrix_{rec_name}.csv"),
              encoding="utf-8")


# ─────────────────────────── Summary ────────────────────────────────

def write_summary(all_ica_results: dict, kurtosis_df: pd.DataFrame,
                  output_dir: str) -> None:
    path = os.path.join(output_dir, "resumen_ica.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Resumen ICA - MIT-BIH Arrhythmia P-Wave\n\n")
        f.write("## Dataset\n")
        f.write("- Fuente: MIT-BIH Arrhythmia Database P-Wave Annotations "
                "(PhysioNet)\n")
        f.write(f"- Registros analizados: {list(all_ica_results.keys())}\n")
        f.write("- Canales por registro: 2 (MLII + V1/V2/V5)\n")
        f.write("- Frecuencia de muestreo: 360 Hz\n")
        f.write(f"- Ventana analizada: {SAMPLE_WINDOW} muestras "
                f"({SAMPLE_WINDOW / 360:.1f} s)\n\n")
        f.write("## Preprocesamiento\n")
        f.write("- StandardScaler por canal (media=0, std=1)\n")
        f.write("- Ventana de 10 segundos desde el inicio del registro\n\n")
        f.write("## Método\n")
        f.write("- FastICA (scikit-learn), 2 componentes, "
                "whitening='unit-variance'\n\n")
        f.write("## Resultados clave\n")
        avg_k_orig = kurtosis_df["kurtosis_original"].abs().mean()
        avg_k_ic = kurtosis_df["kurtosis_ic"].abs().mean()
        f.write(f"- Kurtosis promedio (|valor|) — originales: "
                f"{avg_k_orig:.2f}, ICA: {avg_k_ic:.2f}\n")
        f.write("- Componentes ICA tienden a mayor kurtosis "
                "(mayor no-gaussianidad)\n\n")
        f.write("## Figuras generadas\n")
        for rec in all_ica_results:
            f.write(f"- fig_ica_01_originales_{rec}: Señales originales\n")
            f.write(f"- fig_ica_02_componentes_{rec}: Componentes ICA\n")
            f.write(f"- fig_ica_03_comparacion_{rec}: Original vs ICA\n")
        f.write("- fig_ica_04_kurtosis: Kurtosis comparativa\n")
        f.write("- fig_ica_05_pwave_*: Detalle con anotaciones P-wave\n\n")
        f.write("## Tablas generadas\n")
        f.write("- ica_kurtosis.csv\n")
        f.write("- ica_mixing_matrix_*.csv (por registro)\n")


# ─────────────────────────── Main ───────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICA sobre señales ECG (MIT-BIH P-Wave).")
    parser.add_argument("--data-dir",
                        default=os.path.join("datasets", "physionet-ecg"),
                        help="Directorio con archivos wfdb descargados.")
    parser.add_argument("--output",
                        default=os.path.join("outputs", "ica"),
                        help="Directorio de salida.")
    parser.add_argument("--records", nargs="*", default=None,
                        help="Registros a procesar (default: 100, 119, 207).")
    parser.add_argument("--window", type=int, default=SAMPLE_WINDOW,
                        help="Muestras por ventana (default 3600 = 10s).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _ensure(args.output)
    records = args.records if args.records else DEMO_RECORDS

    all_results = {}

    for rec in records:
        print(f"[ICA] Procesando registro {rec}...")

        try:
            signals, rec_obj = load_record(args.data_dir, rec,
                                           start=0, length=args.window)
        except Exception as e:
            print(f"  ADVERTENCIA: No se pudo cargar {rec}: {e}")
            print(f"  Ejecute primero: python scripts/download_physionet.py")
            continue

        sig_names = rec_obj.sig_name
        fs = rec_obj.fs
        n_channels = signals.shape[1]

        print(f"  Canales: {sig_names}, fs={fs} Hz, "
              f"muestras: {signals.shape[0]}")

        pwave_ann = load_pwave_annotations(args.data_dir, rec)
        if pwave_ann is not None:
            n_pw = len(pwave_ann.sample)
            n_pw_window = np.sum(
                (pwave_ann.sample >= 0) & (pwave_ann.sample < args.window))
            print(f"  Anotaciones P-wave: {n_pw} total, "
                  f"{n_pw_window} en ventana")
        else:
            print("  Sin anotaciones P-wave disponibles")

        S, ica_model, scaler = apply_ica(signals, n_channels, args.seed)

        all_results[rec] = (signals, S, sig_names)

        plot_original_signals(signals, sig_names, fs, rec, args.output,
                              pwave_ann)
        plot_ica_components(S, fs, rec, args.output, pwave_ann)
        plot_comparison(signals, S, sig_names, fs, rec, args.output)
        save_mixing_matrix(ica_model, rec, args.output)

        if pwave_ann is not None:
            plot_pwave_detail(signals, S, sig_names, fs, rec,
                              pwave_ann, args.output)

        print(f"  Figuras generadas para registro {rec}")

    if not all_results:
        print("[ICA] No se procesó ningún registro. "
              "Verifique que los datos estén descargados.")
        return

    print("[ICA] Calculando kurtosis global...")
    kurtosis_df = compute_kurtosis_all(all_results, args.output)
    plot_kurtosis(kurtosis_df, args.output)

    write_summary(all_results, kurtosis_df, args.output)
    print(f"[ICA] Completado. Salidas en: {args.output}")


if __name__ == "__main__":
    main()
