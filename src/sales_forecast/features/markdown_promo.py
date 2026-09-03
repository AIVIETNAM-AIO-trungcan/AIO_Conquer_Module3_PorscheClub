"""Block MarkDown/Promo — Giai đoạn 3.

Quyết định team (docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo
notebooks/01. Preprocessing.ipynb" — GHI ĐÈ quyết định MNAR/flag has_markdown
trước đó): MarkDown1-5 NaN được fillna(0) trực tiếp, KHÔNG còn tạo flag
has_markdown_{i}, khớp notebooks/01. Preprocessing.ipynb Cell 10.
"""

import pandas as pd

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fillna(0) cho MarkDown1-5 — không còn tạo flag has_markdown_{i}."""
    out = df.copy()
    existing = [c for c in MARKDOWN_COLS if c in out.columns]
    if existing:
        out[existing] = out[existing].fillna(0)
    return out
