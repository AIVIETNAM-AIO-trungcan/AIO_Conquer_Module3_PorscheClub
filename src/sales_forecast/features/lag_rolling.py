"""Block Lag/Rolling — Giai đoạn 3.

Hàm thuần: chỉ dùng Weekly_Sales lịch sử của chính (Store, Dept), tính theo
tuần. Dòng đầu chuỗi có lag = NaN (chưa có lịch sử) — KHÔNG fillna 0, vì
NaN ở đây nghĩa là "chưa có lịch sử", khác lag = 0 (CLAUDE.md mục 4 rule 9).
"""

from typing import Iterable, List

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    lags: Iterable[int],
    target_col: str = "Weekly_Sales",
    date_col: str = "Date",
) -> pd.DataFrame:
    """Thêm cột lag_{n}w cho mỗi n trong `lags`, group theo `group_cols`.

    Với mỗi dòng có Date = t, lag_{n}w = Weekly_Sales tại đúng (t - n tuần)
    của cùng nhóm, không phải dòng thứ n trước đó (chuỗi có thể có gap).
    """
    out = df.sort_values(date_col).copy()

    for n in lags:
        col = f"lag_{n}w"
        # Bản sao lịch sử: dịch Date tới trước n tuần để khi merge lại theo
        # Date hiện tại sẽ khớp đúng giá trị của (t - n tuần).
        history = df[group_cols + [date_col, target_col]].copy()
        history[date_col] = history[date_col] + pd.Timedelta(weeks=n)
        history = history.rename(columns={target_col: col})
        out = out.merge(history, on=group_cols + [date_col], how="left")

    return out


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    windows: Iterable[int],
    target_col: str = "Weekly_Sales",
    date_col: str = "Date",
) -> pd.DataFrame:
    """Thêm rolling_mean_{w}w / rolling_std_{w}w, cửa sổ kết thúc TẠI t-1 tuần
    (không bao gồm chính tuần hiện tại, tránh leakage)."""
    out = df.sort_values(date_col).copy()

    for w in windows:
        mean_col = f"rolling_mean_{w}w"
        std_col = f"rolling_std_{w}w"

        out[mean_col] = out.groupby(group_cols)[target_col].transform(
            lambda s, window=w: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        out[std_col] = out.groupby(group_cols)[target_col].transform(
            lambda s, window=w: s.shift(1).rolling(window=window, min_periods=1).std()
        )

    return out
