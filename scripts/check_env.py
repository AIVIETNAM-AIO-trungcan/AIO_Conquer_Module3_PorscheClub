"""Xác thực môi trường trước khi chạy pipeline hoặc dashboard.

Mục đích: cho phép mentor/thành viên mới chạy đúng 1 lệnh để biết môi trường
đã sẵn sàng hay chưa, thay vì tự dò lỗi import từng package.

Cách dùng:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 14)  # exclusive upper bound

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "lightgbm",
    "xgboost",
    "optuna",
    "shap",
    "pandera",
    "yaml",
    "pyarrow",
    "plotly",
    "streamlit",
]

EXPECTED_RAW_FILES = {
    "train.csv": 421_570,
    "test.csv": 115_064,
    "features.csv": 8_190,
    "stores.csv": 45,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"


def check_python_version() -> bool:
    version = sys.version_info[:2]
    ok = MIN_PYTHON <= version < MAX_PYTHON
    status = "OK" if ok else "LOI"
    print(f"[{status}] Python version: {sys.version.split()[0]} "
          f"(yeu cau {MIN_PYTHON[0]}.{MIN_PYTHON[1]} - {MAX_PYTHON[0]}.{MAX_PYTHON[1]-1})")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {pkg} == {version}")
        except ImportError as exc:
            print(f"[LOI] Khong import duoc '{pkg}': {exc}")
            all_ok = False
    return all_ok


def check_raw_data() -> bool:
    if not RAW_DATA_DIR.exists():
        print(f"[LOI] Khong tim thay thu muc data/raw tai: {RAW_DATA_DIR}")
        return False

    all_ok = True
    try:
        import pandas as pd
    except ImportError:
        print("[LOI] Khong the kiem tra data/raw vi pandas chua duoc cai dat")
        return False

    for filename, expected_rows in EXPECTED_RAW_FILES.items():
        path = RAW_DATA_DIR / filename
        if not path.exists():
            print(f"[LOI] Thieu file: data/raw/{filename}")
            all_ok = False
            continue
        n_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1  # trừ header
        if n_rows != expected_rows:
            print(f"[CANH BAO] {filename}: {n_rows} dong (ky vong {expected_rows}) "
                  f"- co the ban dang dung ban data khac voi ban goc du an")
            all_ok = False
        else:
            print(f"[OK] data/raw/{filename}: {n_rows} dong khop ky vong")
    return all_ok


def main() -> int:
    print("=== Kiem tra moi truong Sales & Demand Forecasting ===\n")

    python_ok = check_python_version()
    print()
    packages_ok = check_packages()
    print()
    data_ok = check_raw_data()
    print()

    if python_ok and packages_ok and data_ok:
        print("=== KET QUA: Moi truong san sang. ===")
        return 0

    print("=== KET QUA: Moi truong CHUA san sang, xem chi tiet loi o tren. ===")
    print("Tham khao: docs/06_environment_setup.md muc 7 (Loi thuong gap).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
