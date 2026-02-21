#!/usr/bin/env python3
"""
Orquestador: ejecuta todos los scripts del laboratorio en secuencia.

1. Descarga datos PhysioNet (si faltan)
2. SVD sobre MovieLens 100k
3. t-SNE y UMAP sobre Breast Cancer Wisconsin
4. ICA sobre señales ECG MIT-BIH P-Wave

Uso:
  python scripts/run_all.py
  python scripts/run_all.py --seed 42 --skip-download
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)


def run_script(name: str, args: list[str]) -> bool:
    """Ejecuta un script Python y devuelve True si fue exitoso."""
    script_path = os.path.join(SCRIPTS_DIR, name)
    cmd = [sys.executable, script_path] + args
    print(f"\n{'='*60}")
    print(f"  Ejecutando: {name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n[OK] {name} completado en {elapsed:.1f}s")
        return True
    else:
        print(f"\n[ERROR] {name} falló (código {result.returncode})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta todos los scripts del laboratorio.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Omitir descarga de datos PhysioNet.")
    parser.add_argument("--svd-components", type=int, default=50,
                        help="Componentes SVD (default 50).")
    args = parser.parse_args()

    results = {}
    t_global = time.time()

    # 1. Descarga PhysioNet
    if not args.skip_download:
        ok = run_script("download_physionet.py", [])
        results["download_physionet"] = ok
    else:
        print("\n[SKIP] Descarga de PhysioNet omitida.")
        results["download_physionet"] = "skipped"

    # 2. SVD - MovieLens
    ok = run_script("svd_movielens.py", [
        "--components", str(args.svd_components),
        "--seed", str(args.seed),
    ])
    results["svd_movielens"] = ok

    # 3. t-SNE + UMAP - Breast Cancer
    ok = run_script("tsne_umap_cancer.py", [
        "--seed", str(args.seed),
    ])
    results["tsne_umap_cancer"] = ok

    # 4. ICA - ECG
    ok = run_script("ica_ecg.py", [
        "--seed", str(args.seed),
    ])
    results["ica_ecg"] = ok

    elapsed_total = time.time() - t_global

    # Resumen final
    print(f"\n{'='*60}")
    print("  RESUMEN DE EJECUCION")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "OK" if status is True else ("SKIP" if status == "skipped"
                                            else "FAIL")
        print(f"  [{icon}] {name}")
    print(f"\n  Tiempo total: {elapsed_total:.1f}s")
    print(f"\n  Salidas en:")
    print(f"    outputs/svd/   — SVD (MovieLens)")
    print(f"    outputs/tsne/  — t-SNE (Breast Cancer)")
    print(f"    outputs/umap/  — UMAP (Breast Cancer)")
    print(f"    outputs/ica/   — ICA (ECG P-Wave)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
