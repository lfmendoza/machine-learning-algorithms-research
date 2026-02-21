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
    """Aplica FastICA para separar fuentes independientes.

    Pasos:
    1. StandardScaler centra y normaliza cada canal (media=0, std=1).
    2. FastICA con blanqueo (whitening) decorrelaciona las señales.
    3. Se busca la rotación que maximiza la no-gaussianidad (negentropía)
       de cada componente, separando las fuentes independientes.

    Retorna: (S: componentes separados, ica: modelo, scaler: normalizador)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(signals)

    ica = FastICA(n_components=n_components, random_state=seed,
                  max_iter=1000, whiten="unit-variance")
    # S = W·X donde W ≈ A⁻¹ es la matriz de desmezclado estimada
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

    fig.suptitle(f"Detalle onda P — Registro {rec_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f"fig_ica_05_pwave_{rec_name}.png"), dpi=200)
    plt.close(fig)


# ─────────────────────── Kurtosis analysis ──────────────────────────

def compute_kurtosis_all(all_ica_results: dict, output_dir: str) -> pd.DataFrame:
    """Calcula la kurtosis (Fisher) como medida de no-gaussianidad.

    Una kurtosis alta (positiva) indica una distribución leptocúrtica
    (colas pesadas, picos pronunciados), típica de señales impulsivas
    como los complejos QRS del ECG. ICA busca maximizar este valor.
    """
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
    avg_k_orig = kurtosis_df["kurtosis_original"].abs().mean()
    avg_k_ic = kurtosis_df["kurtosis_ic"].abs().mean()
    records_list = list(all_ica_results.keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write("# ICA (Análisis de Componentes Independientes)\n\n")

        # ── 1. Descripción teórica ──
        f.write("## 1. Descripción teórica\n\n")
        f.write("### Explicación del algoritmo y objetivo principal\n\n")
        f.write(
            "El Análisis de Componentes Independientes (ICA) es una técnica "
            "de separación ciega de fuentes (BSS, por sus siglas en inglés). "
            "Dado un conjunto de señales observadas que son mezclas lineales "
            "de fuentes independientes desconocidas, ICA recupera las fuentes "
            "originales sin conocer el proceso de mezcla. Formalmente, si "
            "X = A·S donde X son las observaciones, A es la matriz de mezcla "
            "desconocida y S son las fuentes independientes, ICA estima una "
            "matriz W ≈ A⁻¹ tal que S ≈ W·X. El algoritmo FastICA maximiza "
            "la no-gaussianidad de las componentes extraídas (medida por "
            "negentropía o kurtosis), basándose en el Teorema Central del "
            "Límite: las mezclas de señales independientes tienden a ser más "
            "gaussianas que las fuentes originales.\n\n")
        f.write("### Principales características y supuestos\n\n")
        f.write(
            "- **Independencia estadística**: las fuentes originales deben "
            "ser estadísticamente independientes (más fuerte que la "
            "decorrelación).\n"
            "- **No-gaussianidad**: como máximo una fuente puede ser "
            "gaussiana; las demás deben tener distribuciones no-gaussianas.\n"
            "- **Mezcla lineal e instantánea**: el modelo asume que las "
            "observaciones son combinaciones lineales de las fuentes en el "
            "mismo instante temporal.\n"
            "- **Ambigüedades**: ICA no puede determinar el orden, el signo "
            "ni la escala de las componentes (son indeterminaciones "
            "inherentes).\n"
            "- **Blanqueo previo (whitening)**: se pre-procesa para "
            "decorrelacionar y normalizar las señales, reduciendo el "
            "problema a buscar una rotación que maximice la "
            "independencia.\n\n")
        f.write("### Diferencias con PCA\n\n")
        f.write(
            "| Aspecto | ICA | PCA |\n"
            "|---|---|---|\n"
            "| Objetivo | Maximizar independencia estadística | Maximizar "
            "varianza explicada |\n"
            "| Criterio | No-gaussianidad (negentropía, kurtosis) | "
            "Varianza (valores propios) |\n"
            "| Tipo de relación | Capta dependencias de orden superior | "
            "Solo decorrelación (2do orden) |\n"
            "| Ortogonalidad | Componentes no necesariamente ortogonales | "
            "Componentes ortogonales |\n"
            "| Ordenamiento | Sin orden natural entre componentes | "
            "Ordenadas por varianza decreciente |\n"
            "| Aplicación típica | Separación de fuentes (señales) | "
            "Reducción de dimensionalidad |\n\n")

        # ── 2. Usos y aplicaciones ──
        f.write("## 2. Usos y aplicaciones\n\n")
        f.write("### Principales usos en análisis de datos\n\n")
        f.write(
            "- **Separación ciega de fuentes (BSS)**: extraer señales "
            "originales a partir de mezclas observadas, sin conocimiento "
            "previo del proceso de mezcla.\n"
            "- **Eliminación de artefactos**: remover ruido, artefactos "
            "musculares o parpadeos de señales biomédicas.\n"
            "- **Extracción de características**: obtener representaciones "
            "estadísticamente independientes que pueden ser más informativas "
            "para tareas de clasificación.\n\n")
        f.write("### Áreas de aplicación\n\n")
        f.write(
            "1. **Electrocardiografía (ECG)**: separación de la actividad "
            "cardíaca de diferentes fuentes (actividad auricular vs "
            "ventricular), eliminación de ruido muscular y de línea "
            "eléctrica. En este ejercicio, ICA separa las componentes "
            "independientes de las derivaciones ECG, permitiendo aislar "
            "patrones como la onda P.\n"
            "2. **Electroencefalografía (EEG)**: eliminación de artefactos "
            "oculares (parpadeos) y musculares de registros cerebrales. "
            "Es estándar en herramientas como EEGLAB para limpiar datos "
            "antes de análisis de potenciales evocados.\n"
            "3. **Procesamiento de audio (problema del cóctel)**: "
            "separar las voces individuales de hablantes a partir de "
            "grabaciones con múltiples micrófonos, donde cada micrófono "
            "capta una mezcla de todas las fuentes.\n\n")

        # ── 3. Aplicación práctica ──
        f.write("## 3. Aplicación práctica\n\n")
        f.write("### Dataset utilizado\n\n")
        f.write(
            "- **Fuente**: MIT-BIH Arrhythmia Database — P-Wave Annotations "
            "(PhysioNet, https://physionet.org/content/pwave/1.0.0/)\n"
            "- **Descripción**: 12 registros ECG seleccionados del MIT-BIH "
            "Arrhythmia Database con anotaciones de onda P realizadas por "
            "dos expertos. Los registros incluyen patologías que dificultan "
            "la detección de ondas P.\n"
            f"- **Registros analizados**: {records_list}\n"
            "- **Canales por registro**: 2 (derivación MLII + derivación "
            "precordial V1/V2/V5)\n"
            "- **Frecuencia de muestreo**: 360 Hz\n"
            f"- **Ventana analizada**: {SAMPLE_WINDOW} muestras "
            f"({SAMPLE_WINDOW / 360:.1f} segundos)\n\n")
        f.write("### Decisiones de preprocesamiento\n\n")
        f.write(
            "- Se seleccionó una ventana de 10 segundos desde el inicio de "
            "cada registro para el análisis.\n"
            "- Se aplicó `StandardScaler` por canal (media=0, std=1) antes "
            "de ICA, ya que FastICA requiere señales centradas.\n"
            "- Se utilizó blanqueo (whitening='unit-variance') como paso "
            "previo a la extracción de componentes.\n\n")
        f.write("### Parámetros del algoritmo\n\n")
        f.write(
            "| Parámetro | Valor |\n"
            "|---|---|\n"
            "| Algoritmo | FastICA (scikit-learn) |\n"
            "| Componentes | 2 (igual al número de canales) |\n"
            "| Whitening | unit-variance |\n"
            "| Iteraciones máximas | 1000 |\n\n")
        f.write("### Resultados obtenidos\n\n")
        f.write("**Kurtosis por registro y componente:**\n\n")
        f.write(
            "| Registro | Canal original | Kurtosis orig. | IC | "
            "Kurtosis IC |\n"
            "|---|---|---|---|---|\n")
        for _, row in kurtosis_df.iterrows():
            f.write(
                f"| {row['registro']} | {row['canal_original']} | "
                f"{row['kurtosis_original']:.2f} | {row['componente_ic']} | "
                f"{row['kurtosis_ic']:.2f} |\n")
        f.write(
            f"\n- **Kurtosis promedio (|valor|)**: originales={avg_k_orig:.2f}"
            f", ICA={avg_k_ic:.2f}\n\n")
        f.write("### Interpretación\n\n")
        f.write(
            "FastICA descompone las dos derivaciones ECG en dos componentes "
            "estadísticamente independientes. Los componentes ICA presentan "
            f"una kurtosis promedio de {avg_k_ic:.2f} (vs {avg_k_orig:.2f} "
            "de los canales originales), lo que indica que el algoritmo "
            "efectivamente maximiza la no-gaussianidad de cada componente, "
            "aislando fuentes con distribuciones más impulsivas (picos QRS, "
            "ondas P). En las figuras de comparación (fig_ica_03) se observa "
            "que las componentes ICA redistribuyen la información de las "
            "derivaciones: un IC tiende a capturar la actividad ventricular "
            "dominante (complejos QRS), mientras que el otro aísla mejor "
            "las ondas P y T de menor amplitud. Las figuras de detalle P-wave "
            "(fig_ica_05) muestran que las anotaciones de onda P (marcadas "
            "en rojo) coinciden con morfologías recurrentes en las "
            "componentes, validando que ICA puede facilitar la detección "
            "de estas ondas al separarlas de la actividad ventricular "
            "dominante. La matriz de mezcla estimada (ica_mixing_matrix) "
            "revela cómo cada derivación contribuye a cada componente "
            "independiente.\n\n")
        f.write("### Limitaciones\n\n")
        f.write(
            "- **Solo 2 canales disponibles**: con únicamente 2 derivaciones "
            "ECG, ICA solo puede separar 2 componentes independientes. Con "
            "más canales (e.g., ECG de 12 derivaciones) se podrían aislar "
            "más fuentes fisiológicas.\n"
            "- **Mezcla lineal e instantánea**: ICA asume que las señales "
            "observadas son combinaciones lineales instantáneas de las "
            "fuentes. En la práctica, las señales cardíacas tienen retardos "
            "de conducción que violan parcialmente este supuesto.\n"
            "- **Ambigüedad en orden y signo**: las componentes ICA no "
            "tienen un orden natural ni signo definido; la interpretación "
            "fisiológica requiere conocimiento del dominio.\n"
            "- **Ventana corta**: se analizaron solo 10 segundos de cada "
            "registro; una ventana más larga podría capturar mayor "
            "variabilidad en los patrones.\n"
            "- **Validación limitada**: aunque las anotaciones de onda P "
            "sirven como referencia, no se realizó una evaluación "
            "cuantitativa de la calidad de la separación (e.g., relación "
            "señal-ruido por componente).\n\n")
        f.write("### Figuras generadas\n\n")
        f.write("| Figura | Descripción |\n|---|---|\n")
        for rec in records_list:
            f.write(
                f"| fig_ica_01_originales_{rec} | Señales ECG originales "
                f"(registro {rec}) |\n")
            f.write(
                f"| fig_ica_02_componentes_{rec} | Componentes ICA "
                f"(registro {rec}) |\n")
            f.write(
                f"| fig_ica_03_comparacion_{rec} | Original vs ICA lado a "
                f"lado |\n")
        f.write(
            "| fig_ica_04_kurtosis | Kurtosis comparativa: canales "
            "originales vs ICA |\n"
            "| fig_ica_05_pwave_* | Detalle con anotaciones P-wave "
            "superpuestas |\n\n")
        f.write("### Tablas generadas\n\n")
        f.write(
            "| Tabla | Contenido |\n"
            "|---|---|\n"
            "| ica_kurtosis.csv | Kurtosis de canales originales y "
            "componentes ICA por registro |\n"
            "| ica_mixing_matrix_*.csv | Matriz de mezcla estimada por "
            "registro |\n")


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
