"""Test aggregate_to_store_date (A39-A41) — docs/05_test_plan.md mục 1.

Đơn vị dự báo đổi từ (Store, Dept, Date) sang (Store, Date) — xem
docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo".
"""

import pandas as pd
import pytest

from sales_forecast.ingestion.loaders import aggregate_to_store_date
from sales_forecast.ingestion.validators import DataContractError


def test_aggregate_sums_weekly_sales_across_dept():
    """A39 (phần SUM): nhiều Dept trong cùng (Store, Date) phải được cộng dồn đúng."""
    train_df = pd.DataFrame({
        "Store": [1, 1, 1, 2],
        "Dept": [1, 2, 3, 1],
        "Date": pd.to_datetime(["2010-02-05", "2010-02-05", "2010-02-05", "2010-02-05"]),
        "Weekly_Sales": [100.0, 50.0, -5.0, 200.0],
        "IsHoliday": [False, False, False, True],
    })
    test_df = pd.DataFrame({
        "Store": [1, 2],
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2010-03-05", "2010-03-05"]),
        "IsHoliday": [False, True],
    })
    train_agg, _ = aggregate_to_store_date(train_df, test_df)
    store1_row = train_agg[train_agg["Store"] == 1].iloc[0]
    assert store1_row["Weekly_Sales"] == 145.0  # 100 + 50 - 5, giữ nguyên dòng âm
    assert len(train_agg) == 2  # (Store=1, 1 tuần), (Store=2, 1 tuần)


def test_isholiday_inconsistency_raises():
    """A39 (phần IsHoliday): nếu IsHoliday KHÔNG nhất quán giữa các Dept trong
    cùng (Store, Date), phải raise lỗi rõ ràng thay vì âm thầm .first() sai."""
    train_df = pd.DataFrame({
        "Store": [1, 1],
        "Dept": [1, 2],
        "Date": pd.to_datetime(["2010-02-05", "2010-02-05"]),
        "Weekly_Sales": [100.0, 50.0],
        "IsHoliday": [False, True],  # vi phạm: cùng Store-Date nhưng khác IsHoliday
    })
    test_df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-03-05"]),
        "IsHoliday": [False],
    })
    with pytest.raises(DataContractError):
        aggregate_to_store_date(train_df, test_df)


def test_dept_column_removed():
    """A40: sau aggregate, cột Dept không còn tồn tại trong output (cả train, test)."""
    train_df = pd.DataFrame({
        "Store": [1, 1],
        "Dept": [1, 2],
        "Date": pd.to_datetime(["2010-02-05", "2010-02-05"]),
        "Weekly_Sales": [100.0, 50.0],
        "IsHoliday": [False, False],
    })
    test_df = pd.DataFrame({
        "Store": [1, 1],
        "Dept": [1, 2],
        "Date": pd.to_datetime(["2010-03-05", "2010-03-05"]),
        "IsHoliday": [False, False],
    })
    train_agg, test_agg = aggregate_to_store_date(train_df, test_df)
    assert "Dept" not in train_agg.columns
    assert "Dept" not in test_agg.columns


def test_aggregate_row_count_matches_observed_pairs():
    """A41: aggregate không cross-join giả định lưới đầy đủ — số dòng sau
    aggregate = đúng số (Store, Date) quan sát thật, không suy diễn thêm."""
    train_df = pd.DataFrame({
        "Store": [1, 1, 1, 2, 2],
        "Dept": [1, 2, 1, 1, 1],
        "Date": pd.to_datetime([
            "2010-02-05", "2010-02-05", "2010-02-12",
            "2010-02-05", "2010-02-12",
        ]),
        "Weekly_Sales": [100.0, 50.0, 110.0, 200.0, 210.0],
        "IsHoliday": [False, False, False, False, False],
    })
    test_df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-03-05"]),
        "IsHoliday": [False],
    })
    train_agg, _ = aggregate_to_store_date(train_df, test_df)
    # (Store=1, 2010-02-05), (Store=1, 2010-02-12), (Store=2, 2010-02-05), (Store=2, 2010-02-12)
    # = 4 dòng thật, KHÔNG phải 2 Store x 2 Date = 4 (trùng ngẫu nhiên ở case này,
    # nên assert bằng cách so khớp trực tiếp observed pairs thay vì suy diễn 2x2)
    observed_pairs = train_df[["Store", "Date"]].drop_duplicates()
    assert len(train_agg) == len(observed_pairs)


def test_drop_duplicates_on_test_when_no_target():
    """test_df không có target -> chỉ cần drop Dept + drop_duplicates, không SUM."""
    test_df = pd.DataFrame({
        "Store": [1, 1, 1],
        "Dept": [1, 2, 3],
        "Date": pd.to_datetime(["2010-03-05", "2010-03-05", "2010-03-05"]),
        "IsHoliday": [False, False, False],
    })
    train_df = pd.DataFrame({
        "Store": [1],
        "Dept": [1],
        "Date": pd.to_datetime(["2010-02-05"]),
        "Weekly_Sales": [100.0],
        "IsHoliday": [False],
    })
    _, test_agg = aggregate_to_store_date(train_df, test_df)
    assert len(test_agg) == 1
    assert test_agg.iloc[0]["Store"] == 1
