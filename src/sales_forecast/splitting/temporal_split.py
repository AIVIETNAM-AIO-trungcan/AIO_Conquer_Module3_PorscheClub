"""Temporal Split cho stage 2 — cắt mốc train/valid/test TRƯỚC feature engineering.

Không dùng random K-Fold (CLAUDE.md mục 4 rule 1). Chạy trước Giai đoạn 3
theo đúng thứ tự docs/02_pipeline_architecture.md.
"""

from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    horizon_weeks: int,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cắt `df` thành (train_window, valid_window) theo `as_of_date = split_date`.

    train_window: Date <= split_date - horizon_weeks tuần
    valid_window: (split_date - horizon_weeks tuần, split_date]

    test_window KHÔNG được tạo ở đây — test.csv thật đã có sẵn khoảng thời
    gian riêng (2012-11-02 → 2013-07-26), không cắt từ train.csv.
    """
    valid_start = split_date - pd.Timedelta(weeks=horizon_weeks)

    train_window = df[df[date_col] <= valid_start].copy()
    valid_window = df[(df[date_col] > valid_start) & (df[date_col] <= split_date)].copy()

    return train_window, valid_window
