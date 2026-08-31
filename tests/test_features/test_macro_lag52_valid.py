"""Test apply_macro_lag52_to_valid — docs/00_decisions.md "Đồng bộ xử lý dữ
liệu theo notebooks/01. Preprocessing.ipynb" (Cell 12).

Hàm mô phỏng "không biết macro hiện tại" cho valid_window: ghi đè macro cols
bằng giá trị cách đây 52 tuần, dùng full_macro_history (gồm cả train) để tra
cứu đúng giá trị của 52 tuần trước, không nhìn tương lai.
"""

import numpy as np
import pandas as pd

from sales_forecast.features.macro import apply_macro_lag52_to_valid


def test_uses_value_52_weeks_before_per_store():
    """Giá trị macro của valid_window phải bằng đúng giá trị macro của chính
    Store đó tại (Date - 52 tuần), lấy từ full_macro_history."""
    full_history = pd.DataFrame({
        "Store": [1, 1],
        "Date": pd.to_datetime(["2011-05-06", "2012-05-04"]),
        "Temperature": [50.0, 99.0],
        "Fuel_Price": [3.0, 5.0],
        "CPI": [210.0, 230.0],
        "Unemployment": [8.0, 6.0],
    })
    valid_df = pd.DataFrame({
        "Store": [1],
        "Date": pd.to_datetime(["2012-05-04"]),
        "Temperature": [99.0],  # giá trị "thật" trước khi ghi đè
        "Fuel_Price": [5.0],
        "CPI": [230.0],
        "Unemployment": [6.0],
    })
    out = apply_macro_lag52_to_valid(valid_df, full_history)
    row = out.iloc[0]
    assert row["Temperature"] == 50.0
    assert row["Fuel_Price"] == 3.0
    assert row["CPI"] == 210.0
    assert row["Unemployment"] == 8.0


def test_does_not_mix_values_between_stores():
    """Giá trị lag52 của Store A không được lẫn với Store B."""
    full_history = pd.DataFrame({
        "Store": [1, 2],
        "Date": pd.to_datetime(["2011-05-06", "2011-05-06"]),
        "Temperature": [50.0, 999.0],
        "Fuel_Price": [3.0, 9.0],
        "CPI": [210.0, 999.0],
        "Unemployment": [8.0, 99.0],
    })
    valid_df = pd.DataFrame({
        "Store": [1],
        "Date": pd.to_datetime(["2012-05-04"]),
        "Temperature": [np.nan],
        "Fuel_Price": [np.nan],
        "CPI": [np.nan],
        "Unemployment": [np.nan],
    })
    out = apply_macro_lag52_to_valid(valid_df, full_history)
    assert out.iloc[0]["CPI"] == 210.0


def test_no_history_52_weeks_before_yields_nan():
    """Nếu không có dữ liệu ở đúng 52 tuần trước, kết quả phải là NaN — không
    tự ý suy diễn/nội suy giá trị khác."""
    full_history = pd.DataFrame({
        "Store": [1],
        "Date": pd.to_datetime(["2011-01-07"]),  # không đúng 52 tuần trước 2012-05-04
        "Temperature": [50.0],
        "Fuel_Price": [3.0],
        "CPI": [210.0],
        "Unemployment": [8.0],
    })
    valid_df = pd.DataFrame({
        "Store": [1],
        "Date": pd.to_datetime(["2012-05-04"]),
        "Temperature": [99.0],
        "Fuel_Price": [5.0],
        "CPI": [230.0],
        "Unemployment": [6.0],
    })
    out = apply_macro_lag52_to_valid(valid_df, full_history)
    assert pd.isna(out.iloc[0]["CPI"])
