"""Block Store Encoding — Giai đoạn 3.

Đổi tên từ store_dept_encoding.py (xem docs/00_decisions.md [2026-08-19]
"Đổi đơn vị dự báo: (Store, Dept, Date) -> (Store, Date)" — Dept không còn
tồn tại từ bước ingestion trở đi).

Giữ Store làm categorical native (LightGBM xử lý trực tiếp,
xem docs/01_ideation.md mục 8) — KHÔNG one-hot để tránh phình chiều
(docs/03_data_io_diagram.md mục 3). Target encoding (fit trên train, áp
dụng cho valid/test) để dành cho giai đoạn model training.
"""

import pandas as pd


def encode_store(df: pd.DataFrame) -> pd.DataFrame:
    """Ép Store thành dtype category, giữ nguyên giá trị."""
    out = df.copy()
    if "Store" in out.columns:
        out["Store"] = out["Store"].astype("category")
    return out
