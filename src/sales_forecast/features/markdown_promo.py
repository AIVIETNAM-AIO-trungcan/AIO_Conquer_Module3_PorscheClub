"""Block MarkDown/Promo — Giai đoạn 3.

MarkDown NaN = "không có chương trình khuyến mãi" (MNAR, xem docs/00_decisions.md),
KHÔNG fillna(0) mù quáng — cần flag has_markdown_{i} tường minh
(CLAUDE.md mục 4 rule 9).
"""

import pandas as pd


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột has_markdown_{i} = True khi MarkDown{i} không NaN.

    Cột MarkDown{i} gốc được giữ NGUYÊN NaN — không fillna ở bước này.
    """
    out = df.copy()
    for i in range(1, 6):
        col = f"MarkDown{i}"
        if col in out.columns:
            out[f"has_markdown_{i}"] = out[col].notna()
    return out
