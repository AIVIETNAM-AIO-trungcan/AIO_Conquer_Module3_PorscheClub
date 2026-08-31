"""Block Macro — Giai đoạn 3.

Join Temperature/Fuel_Price/CPI/Unemployment từ features.csv theo
(Store, Date, IsHoliday). Tái dùng join_features() đã có ở ingestion/loaders.py
thay vì viết lại join (tránh 2 chỗ code join khác nhau cho cùng 1 cặp khóa —
rủi ro lệch logic).

Quyết định team (docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo
notebooks/01. Preprocessing.ipynb" — GHI ĐÈ quyết định "Xử lý CPI/Unemployment
missing" [2026-08-19]): forward-fill theo Store áp dụng cho CẢ 4 CỘT
(Temperature, Fuel_Price, CPI, Unemployment), không chỉ CPI/Unemployment —
khớp notebooks/01. Preprocessing.ipynb Cell 10. Xem thêm
apply_macro_lag52_to_valid() — logic riêng cho valid_window, gọi từ pipeline
orchestration, không phải trong hàm thuần add_macro_features().
"""

from typing import List

import pandas as pd

from sales_forecast.ingestion.loaders import join_features

MACRO_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
FORWARD_FILL_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]


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
    """Join macro columns theo (Store, Date, IsHoliday), sau đó forward-fill
    cả 4 cột macro còn thiếu (Temperature, Fuel_Price, CPI, Unemployment).
    Không nhân bản/mất dòng của `df`."""
    cols = ["Store", "Date", "IsHoliday"] + [c for c in MACRO_COLS if c in features_df.columns]
    joined = join_features(df, features_df[cols])
    joined = _add_macro_forward_fill(joined)
    return joined


def apply_macro_lag52_to_valid(
    valid_df: pd.DataFrame,
    full_macro_history: pd.DataFrame,
    group_col: str = "Store",
    date_col: str = "Date",
    macro_cols: List[str] = MACRO_COLS,
) -> pd.DataFrame:
    """Ghi đè macro cols của valid_window bằng giá trị cách đây 52 tuần.

    Quyết định team (docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo
    notebooks/01. Preprocessing.ipynb" Cell 12): mô phỏng "không biết macro
    hiện tại" khi đánh giá trên valid_window — CHỈ áp dụng cho valid_df,
    train_window vẫn dùng macro thật đồng thời (bất đối xứng có chủ đích, đã
    xác nhận với team, không phải bug). Hàm này KHÔNG được gọi bên trong
    add_macro_features() (hàm đó phải giữ thuần, dùng chung cho mọi tập) —
    trách nhiệm gọi hàm này thuộc về pipeline orchestration, sau khi đã tách
    riêng valid_window.

    `full_macro_history` phải là bảng macro đầy đủ (gồm cả train), vì giá trị
    52 tuần trước của các dòng đầu valid_window nằm trong train_window.
    """
    lookup = full_macro_history[[group_col, date_col] + macro_cols].copy()
    lookup = lookup.rename(columns={c: f"{c}_lag52" for c in macro_cols})
    lookup[date_col] = lookup[date_col] + pd.Timedelta(weeks=52)

    out = valid_df.merge(lookup, on=[group_col, date_col], how="left")
    for col in macro_cols:
        lag_col = f"{col}_lag52"
        out[col] = out[lag_col]
        out = out.drop(columns=[lag_col])
    return out
