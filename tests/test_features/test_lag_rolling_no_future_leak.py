"""Test Lag/Rolling (A6, A42) — docs/05_test_plan.md mục 2.3.

group_cols=["Store"] (không còn Dept) — đơn vị dự báo đã đổi sang (Store, Date),
xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"."""

import pandas as pd

from sales_forecast.features.lag_rolling import add_lag_features


def test_lag_feature_uses_only_past_data(sample_train_aggregated):
    """Giả định A6/A42: với mỗi dòng có Date = t, cột lag_1w phải bằng
    Weekly_Sales của đúng 1 tuần trước đó (Date = t - 7 ngày), KHÔNG được
    bằng giá trị của chính dòng t hay dòng tương lai."""
    df = add_lag_features(sample_train_aggregated, group_cols=["Store"], lags=[1])
    row = df[(df.Store == 1) & (df.Date == "2010-02-12")].iloc[0]
    expected = sample_train_aggregated[
        (sample_train_aggregated.Store == 1) & (sample_train_aggregated.Date == "2010-02-05")
    ]["Weekly_Sales"].iloc[0]
    assert row["lag_1w"] == expected


def test_first_observation_has_nan_lag_not_zero(sample_train_aggregated):
    """Giả định A6/A42 (liên quan A8): dòng đầu tiên của 1 chuỗi phải có lag = NaN,
    KHÔNG được điền 0 ở bước tạo feature (fillna là quyết định tường minh ở bước sau)."""
    df = add_lag_features(sample_train_aggregated, group_cols=["Store"], lags=[1])
    first_row = df[df.Store == 1].sort_values("Date").iloc[0]
    assert pd.isna(first_row["lag_1w"])
