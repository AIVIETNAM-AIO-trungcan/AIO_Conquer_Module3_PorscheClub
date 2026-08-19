"""Pytest fixtures cho test suite.

Dữ liệu giả lập nhỏ, có chủ đích (sales âm, cold-start, MarkDown NaN) —
theo đúng skeleton ở docs/05_test_plan.md mục 2.1, không dùng data/raw/ thật.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_train():
    """Giả lập train.csv thu nhỏ: 2 Store, 2 Dept, 6 tuần, có 1 dòng sales âm."""
    return pd.DataFrame({
        "Store": [1, 1, 1, 1, 2, 2],
        "Dept": [1, 1, 1, 1, 1, 1],
        "Date": pd.to_datetime([
            "2010-02-05", "2010-02-12", "2010-02-19", "2010-02-26",
            "2010-02-05", "2010-02-12",
        ]),
        "Weekly_Sales": [100.0, 120.0, -5.0, 130.0, 200.0, 210.0],
        "IsHoliday": [False, True, False, False, False, True],
    })


@pytest.fixture
def sample_test():
    """(Store=3, Dept=1) chưa từng xuất hiện trong sample_train -> mô phỏng cold-start."""
    return pd.DataFrame({
        "Store": [1, 3],
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2010-03-05", "2010-03-05"]),
        "IsHoliday": [False, False],
    })


@pytest.fixture
def sample_features():
    """features.csv giả lập: MarkDown NaN có chủ đích ở 1 dòng."""
    return pd.DataFrame({
        "Store": [1, 1, 2],
        "Date": pd.to_datetime(["2010-02-05", "2010-02-12", "2010-02-05"]),
        "Temperature": [42.3, 38.5, 40.0],
        "Fuel_Price": [2.5, 2.6, 2.5],
        "MarkDown1": [None, 500.0, None],
        "CPI": [211.1, 211.2, 210.0],
        "Unemployment": [8.1, 8.1, 7.9],
        "IsHoliday": [False, True, False],
    })


@pytest.fixture
def sample_train_aggregated():
    """train.csv giả lập SAU aggregate_to_store_date (Store, Date) — không còn
    Dept. Tương ứng với sample_train sau khi SUM theo Dept:
    (Store=1, 2010-02-05): 100.0; (Store=1, 2010-02-12): 120.0;
    (Store=1, 2010-02-19): -5.0; (Store=1, 2010-02-26): 130.0;
    (Store=2, 2010-02-05): 200.0; (Store=2, 2010-02-12): 210.0.
    Xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"."""
    return pd.DataFrame({
        "Store": [1, 1, 1, 1, 2, 2],
        "Date": pd.to_datetime([
            "2010-02-05", "2010-02-12", "2010-02-19", "2010-02-26",
            "2010-02-05", "2010-02-12",
        ]),
        "Weekly_Sales": [100.0, 120.0, -5.0, 130.0, 200.0, 210.0],
        "IsHoliday": [False, True, False, False, False, True],
    })


@pytest.fixture
def sample_test_aggregated():
    """test.csv giả lập SAU aggregate_to_store_date — Store=3 hoàn toàn mới
    (không có trong sample_train_aggregated) -> mô phỏng cold-start Store-level."""
    return pd.DataFrame({
        "Store": [1, 3],
        "Date": pd.to_datetime(["2010-03-05", "2010-03-05"]),
        "IsHoliday": [False, False],
    })


@pytest.fixture
def sample_stores():
    """stores.csv giả lập."""
    return pd.DataFrame({
        "Store": [1, 2, 3],
        "Type": ["A", "B", "A"],
        "Size": [150000, 120000, 160000],
    })
