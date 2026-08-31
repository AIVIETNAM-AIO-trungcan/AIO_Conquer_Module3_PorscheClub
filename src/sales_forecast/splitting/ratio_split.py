"""Split theo tỷ lệ số ngày duy nhất — Giai đoạn 2 (thay thế cục bộ).

Quyết định team (docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo
notebooks/01. Preprocessing.ipynb" Cell 12): thay vì cắt theo mốc ngày cố
định (temporal_split), chia train/valid theo tỷ lệ 2/3 số ngày duy nhất của
phần dữ liệu đã có target — khớp notebooks/01. Preprocessing.ipynb.

Phạm vi áp dụng: CHỈ pipelines/run_train_baseline.py — không thay thế
temporal_split() ở tầm kiến trúc chung (docs/02_pipeline_architecture.md,
CLAUDE.md invariant #1 không đổi).
"""

from typing import Tuple

import pandas as pd


def split_by_date_ratio(
    df: pd.DataFrame, ratio: float = 2 / 3, date_col: str = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chia `df` thành (train_window, valid_window) theo tỷ lệ số ngày duy nhất.

    `split_idx = int(số ngày duy nhất * ratio)`; mốc chia = ngày thứ
    `split_idx` trong danh sách ngày duy nhất đã sắp xếp tăng dần.
    train_window: Date < mốc chia. valid_window: Date >= mốc chia.

    Không random — vẫn giữ đúng thứ tự thời gian (invariant #1 CLAUDE.md:
    không dùng random K-Fold cho time-series).
    """
    unique_dates = sorted(df[date_col].unique())
    split_idx = int(len(unique_dates) * ratio)
    val_split_date = unique_dates[split_idx]

    train_window = df[df[date_col] < val_split_date].copy()
    valid_window = df[df[date_col] >= val_split_date].copy()

    return train_window, valid_window
