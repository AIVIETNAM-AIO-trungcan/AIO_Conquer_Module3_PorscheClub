"""Test forward-fill macro (A36-A38) — docs/05_test_plan.md mục 1.

Bối cảnh gốc: features.csv thật thiếu CPI/Unemployment đúng 13 tuần cuối
test_window ở mọi Store, do độ trễ công bố macro index — xem
docs/00_decisions.md [2026-08-19] "Xử lý CPI/Unemployment missing ở đuôi
test_window". Quyết định team mới nhất ("Đồng bộ xử lý dữ liệu theo
notebooks/01. Preprocessing.ipynb") mở rộng forward-fill sang CẢ 4 CỘT
(Temperature, Fuel_Price, CPI, Unemployment) — các fixture dưới đây có đủ 4
cột để khớp FORWARD_FILL_COLS mới; test case A36-A38 vẫn giữ nguyên tinh
thần kiểm tra qua CPI (đại diện), thêm test riêng cho Temperature/Fuel_Price.
"""

import numpy as np
import pandas as pd

from sales_forecast.features.macro import _add_macro_forward_fill


def test_forward_fill_does_not_overwrite_existing_values():
    """A36: dòng đã có giá trị CPI/Unemployment thật không bị forward-fill ghi đè."""
    df = pd.DataFrame({
        "Store": [1, 1, 1],
        "Date": pd.to_datetime(["2013-04-19", "2013-04-26", "2013-05-03"]),
        "Temperature": [55.1, 56.2, np.nan],
        "Fuel_Price": [3.5, 3.6, np.nan],
        "CPI": [220.1, 220.2, np.nan],
        "Unemployment": [7.5, 7.4, np.nan],
    })
    out = _add_macro_forward_fill(df)
    # 2 dòng đầu vốn đã có giá trị thật -> giữ nguyên, không bị đụng vào
    assert out.loc[out["Date"] == "2013-04-19", "CPI"].iloc[0] == 220.1
    assert out.loc[out["Date"] == "2013-04-26", "CPI"].iloc[0] == 220.2
    assert not out.loc[out["Date"] == "2013-04-19", "cpi_is_forward_filled"].iloc[0]
    assert not out.loc[out["Date"] == "2013-04-26", "cpi_is_forward_filled"].iloc[0]


def test_forward_fill_uses_last_known_value_per_store():
    """A37: forward-fill dùng đúng giá trị công bố gần nhất theo TỪNG Store,
    không lẫn giá trị giữa các Store khác nhau."""
    df = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Date": pd.to_datetime([
            "2013-04-26", "2013-05-03",
            "2013-04-26", "2013-05-03",
        ]),
        "Temperature": [60.0, np.nan, 70.0, np.nan],
        "Fuel_Price": [3.5, np.nan, 3.8, np.nan],
        "CPI": [220.2, np.nan, 300.0, np.nan],
        "Unemployment": [7.4, np.nan, 9.0, np.nan],
    })
    out = _add_macro_forward_fill(df)
    store1_filled = out[(out["Store"] == 1) & (out["Date"] == "2013-05-03")]
    store2_filled = out[(out["Store"] == 2) & (out["Date"] == "2013-05-03")]
    assert store1_filled["CPI"].iloc[0] == 220.2
    assert store2_filled["CPI"].iloc[0] == 300.0  # không bị lẫn giá trị của Store 1


def test_forward_fill_flag_matches_actually_filled_rows():
    """A38: flag chỉ True đúng ở các dòng thực sự được điền, False ở mọi dòng khác."""
    df = pd.DataFrame({
        "Store": [1, 1, 1],
        "Date": pd.to_datetime(["2013-04-19", "2013-04-26", "2013-05-03"]),
        "Temperature": [55.1, 56.2, np.nan],
        "Fuel_Price": [3.5, 3.6, np.nan],
        "CPI": [220.1, 220.2, np.nan],
        "Unemployment": [7.5, 7.4, np.nan],
    })
    out = _add_macro_forward_fill(df)
    flags = out.set_index("Date")["cpi_is_forward_filled"]
    assert flags.loc[pd.Timestamp("2013-05-03")] == True
    assert flags.loc[pd.Timestamp("2013-04-19")] == False
    assert flags.loc[pd.Timestamp("2013-04-26")] == False


def test_forward_fill_no_history_leaves_nan_and_flag_false():
    """Ca biên: Store hoàn toàn không có giá trị CPI nào (toàn NaN) -> ffill
    không điền được gì, flag vẫn False (không throw lỗi, không đánh dấu sai)."""
    df = pd.DataFrame({
        "Store": [9, 9],
        "Date": pd.to_datetime(["2013-05-03", "2013-05-10"]),
        "Temperature": [np.nan, np.nan],
        "Fuel_Price": [np.nan, np.nan],
        "CPI": [np.nan, np.nan],
        "Unemployment": [np.nan, np.nan],
    })
    out = _add_macro_forward_fill(df)
    assert out["CPI"].isna().all()
    assert (out["cpi_is_forward_filled"] == False).all()


def test_forward_fill_applies_to_temperature_and_fuel_price():
    """Mở rộng theo quyết định team mới nhất: Temperature/Fuel_Price giờ cũng
    được forward-fill (trước đây cố tình loại trừ vì đã xác nhận đủ dữ liệu
    thật) — xem docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo
    notebooks/01. Preprocessing.ipynb"."""
    df = pd.DataFrame({
        "Store": [1, 1],
        "Date": pd.to_datetime(["2013-04-26", "2013-05-03"]),
        "Temperature": [60.0, np.nan],
        "Fuel_Price": [3.5, np.nan],
        "CPI": [220.2, np.nan],
        "Unemployment": [7.4, np.nan],
    })
    out = _add_macro_forward_fill(df)
    last_row = out[out["Date"] == "2013-05-03"].iloc[0]
    assert last_row["Temperature"] == 60.0
    assert last_row["Fuel_Price"] == 3.5
    assert last_row["temperature_is_forward_filled"] == True
    assert last_row["fuel_price_is_forward_filled"] == True
