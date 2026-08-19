"""Block Macro — Giai đoạn 3.

Join Temperature/Fuel_Price/CPI/Unemployment từ features.csv theo (Store, Date).
Tái dùng join_features() đã có ở ingestion/loaders.py thay vì viết lại join
(tránh 2 chỗ code join khác nhau cho cùng 1 cặp khóa — rủi ro lệch logic).

Ghi chú quan trọng (xem docs/00_decisions.md [2026-08-19] "Xử lý CPI/Unemployment
missing ở đuôi test_window"): features.csv đã khảo sát thực tế phủ hết
test_window (đến 2013-07-26) — Temperature/Fuel_Price/MarkDown gần như đầy đủ,
đây LÀ biến ngoại sinh quan sát được tại thời điểm t (given), không cần tự dự
báo. Riêng CPI/Unemployment thiếu ở đuôi test_window do độ trễ công bố macro
index thật (không phải lỗi ngẫu nhiên) — xử lý bằng forward-fill theo Store,
xem _add_macro_forward_fill().
"""

import pandas as pd

from sales_forecast.ingestion.loaders import join_features

MACRO_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
FORWARD_FILL_COLS = ["CPI", "Unemployment"]


def _add_macro_forward_fill(
    df: pd.DataFrame, group_col: str = "Store", date_col: str = "Date"
) -> pd.DataFrame:
    """Forward-fill CPI/Unemployment theo từng Store, dùng giá trị công bố gần
    nhất đã biết trước đó (không nhìn tương lai — chỉ ffill, không bfill).

    CPI/Unemployment là chỉ số kinh tế vĩ mô biến động rất chậm theo tháng/quý,
    nên forward-fill hợp lý hơn fillna(0) (sai đơn vị) hay fillna(mean) (che
    giấu xu hướng thật). Không hard-code cửa sổ thời gian cụ thể (vd. "13 tuần
    cuối") — ffill tự nhiên chỉ điền đúng chỗ có NaN, nhất quán CLAUDE.md mục 4
    rule "không hard-code tham số trong src/". Chỉ đánh dấu flag cho dòng THỰC
    SỰ được điền (trước đó là NaN), không đánh dấu nhầm dòng vốn đã có giá trị
    thật — cùng triết lý `has_markdown` đã chốt: không fillna mù quáng, luôn có
    flag tường minh phân biệt "giá trị thật" và "giá trị suy ra do thiếu dữ liệu".
    """
    out = df.sort_values([group_col, date_col]).copy()
    for col in FORWARD_FILL_COLS:
        flag_col = f"{col.lower()}_is_forward_filled"
        was_na = out[col].isna()
        out[col] = out.groupby(group_col)[col].transform(lambda s: s.ffill())
        out[flag_col] = was_na & out[col].notna()
    return out.sort_index()


def add_macro_features(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Join macro columns theo (Store, Date), sau đó forward-fill CPI/Unemployment
    còn thiếu ở đuôi test_window. Không nhân bản/mất dòng của `df`."""
    cols = ["Store", "Date"] + [c for c in MACRO_COLS if c in features_df.columns]
    joined = join_features(df, features_df[cols])
    joined = _add_macro_forward_fill(joined)
    return joined
