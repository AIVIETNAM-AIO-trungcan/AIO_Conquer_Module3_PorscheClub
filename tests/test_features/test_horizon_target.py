"""Test add_horizon_targets (A53) — docs/05_test_plan.md mục 1.

Khớp notebooks/viet/multi_step/direct_way/*.ipynb Cell 'cell-targets' —
target_t+{h} = Weekly_Sales sau đúng h tuần CÙNG group (Store).
"""

import numpy as np
import pandas as pd

from sales_forecast.features.horizon_target import add_horizon_targets


def _make_df() -> pd.DataFrame:
    dates = pd.date_range("2012-01-06", periods=5, freq="7D")
    return pd.DataFrame({
        "Store": [1, 1, 1, 1, 1] + [2, 2, 2, 2, 2],
        "Date": list(dates) * 2,
        "Weekly_Sales": [100.0, 110.0, 120.0, 130.0, 140.0, 200.0, 210.0, 220.0, 230.0, 240.0],
    })


def test_target_shift_matches_future_value_same_group():
    """A53: target_t+1 tại dòng Date=t phải bằng Weekly_Sales tại Date=t+1
    CÙNG Store, target_t+2 bằng Weekly_Sales tại Date=t+2, v.v."""
    df = _make_df()
    out = add_horizon_targets(df, horizon=3)
    row = out[(out["Store"] == 1) & (out["Date"] == pd.Timestamp("2012-01-06"))].iloc[0]
    assert row["target_t+1"] == 110.0
    assert row["target_t+2"] == 120.0
    assert row["target_t+3"] == 130.0


def test_last_rows_of_group_are_nan_not_leaked_from_other_group():
    """A53: dòng cuối mỗi group không đủ h tuần tương lai -> NaN, KHÔNG được
    lấy nhầm giá trị từ Store khác (bất biến chống leakage-chéo-group)."""
    df = _make_df()
    out = add_horizon_targets(df, horizon=3)
    last_row_store1 = out[(out["Store"] == 1) & (out["Date"] == pd.Timestamp("2012-02-03"))].iloc[0]
    assert pd.isna(last_row_store1["target_t+1"])
    assert pd.isna(last_row_store1["target_t+2"])
    assert pd.isna(last_row_store1["target_t+3"])


def test_does_not_mutate_original_dataframe():
    """A53: hàm trả DataFrame mới, không sửa df gốc (hàm thuần)."""
    df = _make_df()
    original_cols = list(df.columns)
    add_horizon_targets(df, horizon=2)
    assert list(df.columns) == original_cols


def test_works_without_presorted_input():
    """A53: hàm tự sort theo (group_col, date_col), không giả định caller
    đã sort — kết quả đúng dù input bị xáo trộn thứ tự."""
    df = _make_df().sample(frac=1, random_state=1).reset_index(drop=True)
    out = add_horizon_targets(df, horizon=1)
    row = out[(out["Store"] == 2) & (out["Date"] == pd.Timestamp("2012-01-13"))].iloc[0]
    assert row["target_t+1"] == 220.0
