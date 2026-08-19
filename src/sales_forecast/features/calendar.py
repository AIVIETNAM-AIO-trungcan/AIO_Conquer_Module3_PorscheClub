"""Block Calendar — Giai đoạn 3. Hàm thuần, chỉ dùng cột Date."""

import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Thêm month, quarter, day_of_week, week_of_year từ `date_col`."""
    out = df.copy()
    dt = out[date_col].dt
    out["month"] = dt.month
    out["quarter"] = dt.quarter
    out["day_of_week"] = dt.dayofweek
    out["week_of_year"] = dt.isocalendar().week.astype(int)
    return out
