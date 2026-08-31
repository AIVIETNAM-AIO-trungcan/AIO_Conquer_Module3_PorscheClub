"""Test split_by_date_ratio — docs/00_decisions.md "Đồng bộ xử lý dữ liệu
theo notebooks/01. Preprocessing.ipynb" Cell 12.

Chỉ áp dụng cục bộ cho pipelines/run_train_baseline.py — temporal_split()
(mốc ngày cố định) vẫn là cơ chế split kiến trúc chung.
"""

import pandas as pd

from sales_forecast.splitting.ratio_split import split_by_date_ratio


def _make_df(n_weeks: int) -> pd.DataFrame:
    dates = pd.date_range("2012-01-06", periods=n_weeks, freq="7D")
    return pd.DataFrame({"Store": [1] * n_weeks, "Date": dates, "Weekly_Sales": range(n_weeks)})


def test_split_ratio_2_3_gives_correct_sizes():
    """9 ngày duy nhất, ratio=2/3 -> split_idx=6 -> train có 6 ngày đầu, valid có 3 ngày cuối."""
    df = _make_df(9)
    train_w, valid_w = split_by_date_ratio(df, ratio=2 / 3)
    assert len(train_w) == 6
    assert len(valid_w) == 3


def test_train_window_before_valid_window():
    """Không chồng lấn ngày, train luôn trước valid (không random)."""
    df = _make_df(12)
    train_w, valid_w = split_by_date_ratio(df, ratio=2 / 3)
    assert train_w["Date"].max() < valid_w["Date"].min()
    overlap = pd.merge(train_w, valid_w, on=["Store", "Date"], how="inner")
    assert len(overlap) == 0


def test_split_uses_unique_dates_not_row_count():
    """Nhiều Store cùng ngày không làm lệch mốc chia — mốc tính theo số NGÀY
    duy nhất, không phải số dòng."""
    dates = pd.date_range("2012-01-06", periods=9, freq="7D")
    df = pd.DataFrame({
        "Store": [1, 2] * 9,
        "Date": sorted(list(dates) * 2),
        "Weekly_Sales": range(18),
    })
    train_w, valid_w = split_by_date_ratio(df, ratio=2 / 3)
    assert train_w["Date"].nunique() == 6
    assert valid_w["Date"].nunique() == 3
    assert len(train_w) == 12  # 6 ngày x 2 Store
    assert len(valid_w) == 6  # 3 ngày x 2 Store
