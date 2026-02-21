#!/usr/bin/env python3
"""
Descarga los 12 registros ECG del MIT-BIH Arrhythmia Database P-Wave
Annotations desde PhysioNet usando la librería wfdb.

Archivos por registro: .hea (header), .dat (señal), .pwave (anotaciones P)

Uso:
  python scripts/download_physionet.py
  python scripts/download_physionet.py --output datasets/physionet-ecg
"""
from __future__ import annotations

import argparse
import os

import wfdb

RECORDS = ["100", "101", "103", "106", "117", "119",
           "122", "207", "214", "222", "223", "231"]

DB_NAME = "pwave"
EXPECTED_EXTENSIONS = (".hea", ".dat", ".pwave")


def records_present(dl_dir: str) -> list[str]:
    """Return list of record names that already have all three files locally."""
    complete = []
    for rec in RECORDS:
        if all(os.path.isfile(os.path.join(dl_dir, f"{rec}{ext}"))
               for ext in EXPECTED_EXTENSIONS):
            complete.append(rec)
    return complete


def download(dl_dir: str) -> None:
    os.makedirs(dl_dir, exist_ok=True)

    already = records_present(dl_dir)
    if len(already) == len(RECORDS):
        print(f"Todos los {len(RECORDS)} registros ya están en {dl_dir}. "
              "Nada que descargar.")
        return

    missing = [r for r in RECORDS if r not in already]
    print(f"Registros completos: {len(already)}/{len(RECORDS)}")
    print(f"Descargando {len(missing)} registros faltantes: {missing}")

    wfdb.dl_database(DB_NAME, dl_dir=dl_dir, records=missing)

    final = records_present(dl_dir)
    print(f"\nDescarga finalizada. Registros completos: {len(final)}/{len(RECORDS)}")

    still_missing = [r for r in RECORDS if r not in final]
    if still_missing:
        print(f"ADVERTENCIA: faltan archivos para: {still_missing}")
    else:
        print("Todos los registros descargados correctamente.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga registros ECG P-Wave desde PhysioNet.")
    parser.add_argument(
        "--output", default=os.path.join("datasets", "physionet-ecg"),
        help="Directorio destino (default: datasets/physionet-ecg)")
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
