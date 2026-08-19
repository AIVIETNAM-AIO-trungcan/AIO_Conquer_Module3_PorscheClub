"""Test cold-start Store-level (A8 DEPRECATED, A43) — docs/05_test_plan.md mục 2.4.

A8 gốc (11 cặp Store-Dept cold-start) không còn áp dụng — đơn vị dự báo đã đổi
sang (Store, Date), đã xác nhận KHÔNG có Store nào cold-start trên dữ liệu thật
hiện tại (cả 45 Store đều có trong train). Test này giữ lại has_history như một
bất biến phòng thủ (defensive) — nếu tương lai xuất hiện Store hoàn toàn mới,
flag vẫn phải hoạt động đúng. Xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị
dự báo"."""

import pandas as pd

from sales_forecast.features.pipeline import build_feature_matrix


def test_cold_start_store_flagged(sample_train_aggregated, sample_test_aggregated, sample_features):
    """Giả định A43: Store=3 không có trong train phải được đánh dấu
    has_history=False, KHÔNG được crash pipeline và KHÔNG được âm thầm dùng
    lag từ Store khác."""
    fm = build_feature_matrix(sample_train_aggregated, sample_test_aggregated, sample_features)
    cold_row = fm[fm.Store == 3]
    assert len(cold_row) == 1
    assert cold_row["has_history"].iloc[0] == False
    assert pd.isna(cold_row["lag_1w"].iloc[0])


def test_known_store_has_history_true(sample_train_aggregated, sample_test_aggregated, sample_features):
    """Store=1 có trong train phải được đánh dấu has_history=True."""
    fm = build_feature_matrix(sample_train_aggregated, sample_test_aggregated, sample_features)
    known_row = fm[(fm.Store == 1) & (fm.Date == pd.Timestamp("2010-03-05"))]
    assert known_row["has_history"].iloc[0] == True
