"""Test Temporal Split (A5) — docs/05_test_plan.md mục 2.3.

Input dùng fixture aggregated (Store, Date) — temporal_split nhận dữ liệu ĐÃ
qua aggregate_to_store_date (xem docs/00_decisions.md [2026-08-19]
"Đổi đơn vị dự báo"), không còn Dept."""

import pandas as pd

from sales_forecast.splitting.temporal_split import temporal_split


def test_train_window_before_valid_window(sample_train_aggregated):
    """Giả định A5: mọi Date trong train_window phải < mọi Date trong valid_window."""
    split_date = pd.Timestamp("2010-02-19")
    train_w, valid_w = temporal_split(sample_train_aggregated, split_date=split_date, horizon_weeks=1)
    assert train_w["Date"].max() < valid_w["Date"].min()


def test_valid_window_does_not_leak_into_train(sample_train_aggregated):
    """Giả định A5: không có bản ghi nào xuất hiện ở cả train_window và valid_window."""
    split_date = pd.Timestamp("2010-02-19")
    train_w, valid_w = temporal_split(sample_train_aggregated, split_date=split_date, horizon_weeks=1)
    overlap = pd.merge(train_w, valid_w, on=["Store", "Date"], how="inner")
    assert len(overlap) == 0
