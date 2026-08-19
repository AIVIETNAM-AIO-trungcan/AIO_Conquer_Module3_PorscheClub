"""Test bất biến chống leakage của cơ chế buffer nối train_window/valid_window
(A46) — docs/05_test_plan.md mục 1.

Bối cảnh: build_feature_matrix (src/sales_forecast/features/pipeline.py) gộp
train_window + valid_window(không target) thành 1 bảng liên tục theo thời gian
TRƯỚC khi tính lag/rolling, để tránh NaN giả tạo ở các dòng đầu valid_window
(vốn có đủ lịch sử thật từ cuối train_window). File test này chứng minh cơ chế
gộp bảng đó KHÔNG rò rỉ dữ liệu tương lai vào train_window — xem
docs/00_decisions.md [2026-08-19] "Chính thức hóa cơ chế buffer nối
train_window/valid_window cho Lag/Rolling/Macro".
"""

import pandas as pd
from pandas.testing import assert_series_equal

from sales_forecast.features.lag_rolling import add_lag_features, add_rolling_features


def _make_continuous_store_series(n_weeks: int, start_sales: float = 100.0) -> pd.DataFrame:
    """1 Store, n_weeks tuần liên tục, Weekly_Sales tăng dần 10 mỗi tuần —
    dễ nhận biết đúng/sai giá trị lag/rolling."""
    dates = pd.date_range("2012-01-06", periods=n_weeks, freq="7D")
    sales = [start_sales + 10.0 * i for i in range(n_weeks)]
    return pd.DataFrame({
        "Store": [1] * n_weeks,
        "Date": dates,
        "Weekly_Sales": sales,
    })


def test_train_window_lag_unchanged_when_valid_window_appended():
    """A46 (bất biến quan trọng nhất): giá trị lag_1w của các dòng thuộc
    train_window phải giống hệt nhau dù tính trên train_window đơn lẻ hay
    trên bảng gộp train_window + valid_window(không target) — chứng minh việc
    gộp thêm valid vào không rò rỉ ngược làm thay đổi feature của train."""
    full = _make_continuous_store_series(n_weeks=8)
    train_window = full.iloc[:5].copy()  # 5 tuần đầu
    valid_window = full.iloc[5:].copy()  # 3 tuần cuối, sẽ bỏ target khi gộp

    # Tính lag CHỈ trên train_window (không có "buffer")
    lag_train_alone = add_lag_features(train_window, group_cols=["Store"], lags=[1])

    # Tính lag trên bảng gộp train_window + valid_window(không target)
    valid_no_target = valid_window.drop(columns=["Weekly_Sales"])
    combined = pd.concat([train_window, valid_no_target], ignore_index=True, sort=False)
    lag_combined = add_lag_features(combined, group_cols=["Store"], lags=[1])
    lag_combined_train_part = lag_combined[lag_combined["Date"].isin(train_window["Date"])]

    merged = lag_train_alone.merge(
        lag_combined_train_part, on=["Store", "Date"], suffixes=("_alone", "_combined")
    )
    # so sánh bằng assert_series_equal (không phải ==) vì NaN == NaN trả về
    # False trong pandas, trong khi ở đây NaN xuất hiện đúng vị trí (dòng đầu
    # chuỗi, chưa có lịch sử) ở cả 2 phía và PHẢI được coi là bằng nhau
    assert_series_equal(
        merged["lag_1w_alone"], merged["lag_1w_combined"],
        check_names=False,
        obj="lag_1w: train_window đơn lẻ vs. train_window trong bảng gộp",
    )


def test_valid_window_first_row_gets_lag_from_train_tail_not_nan():
    """A46: dòng đầu tiên của valid_window (sau khi bỏ target, gộp vào bảng)
    phải nhận đúng lag_1w = Weekly_Sales thật của dòng cuối train_window liền
    trước — KHÔNG được NaN oan (đây chính là vấn đề buffer window team nêu ra)."""
    full = _make_continuous_store_series(n_weeks=6)
    train_window = full.iloc[:4].copy()
    valid_window = full.iloc[4:].copy()

    valid_no_target = valid_window.drop(columns=["Weekly_Sales"])
    combined = pd.concat([train_window, valid_no_target], ignore_index=True, sort=False)
    lag_combined = add_lag_features(combined, group_cols=["Store"], lags=[1])

    first_valid_date = valid_window["Date"].min()
    first_valid_row = lag_combined[lag_combined["Date"] == first_valid_date].iloc[0]
    expected = train_window[train_window["Date"] == train_window["Date"].max()]["Weekly_Sales"].iloc[0]

    assert not pd.isna(first_valid_row["lag_1w"]), (
        "Dòng đầu valid_window không được NaN oan — phải lấy được lag từ "
        "dữ liệu thật cuối train_window"
    )
    assert first_valid_row["lag_1w"] == expected


def test_valid_window_second_row_onward_naturally_nan():
    """A46: dòng thứ 2 trở đi trong valid_window (chưa biết target) phải có
    lag_1w = NaN — đây là GIỚI HẠN TỰ NHIÊN của direct multi-step forecasting
    (valid/test không thể tự cung cấp lag cho chính nó), KHÔNG PHẢI bug.
    Test này tài liệu hóa hành vi để tránh hiểu nhầm là lỗi sau này."""
    full = _make_continuous_store_series(n_weeks=6)
    train_window = full.iloc[:4].copy()
    valid_window = full.iloc[4:].copy()  # 2 tuần: index 4, 5

    valid_no_target = valid_window.drop(columns=["Weekly_Sales"])
    combined = pd.concat([train_window, valid_no_target], ignore_index=True, sort=False)
    lag_combined = add_lag_features(combined, group_cols=["Store"], lags=[1])

    second_valid_date = valid_window["Date"].iloc[1]
    second_valid_row = lag_combined[lag_combined["Date"] == second_valid_date].iloc[0]

    assert pd.isna(second_valid_row["lag_1w"]), (
        "Dòng thứ 2 trở đi của valid_window PHẢI NaN vì tuần liền trước nó "
        "(dòng đầu valid_window) chưa có Weekly_Sales thật để lag — giới hạn "
        "tự nhiên, không phải lỗi"
    )


def test_train_window_rolling_unchanged_when_valid_window_appended():
    """A46: tương tự lag, rolling_mean của train_window phải bất biến dù tính
    đơn lẻ hay trên bảng gộp — chứng minh add_rolling_features cũng không rò
    rỉ ngược."""
    full = _make_continuous_store_series(n_weeks=8)
    train_window = full.iloc[:5].copy()
    valid_window = full.iloc[5:].copy()

    rolling_train_alone = add_rolling_features(train_window, group_cols=["Store"], windows=[2])

    valid_no_target = valid_window.drop(columns=["Weekly_Sales"])
    combined = pd.concat([train_window, valid_no_target], ignore_index=True, sort=False)
    rolling_combined = add_rolling_features(combined, group_cols=["Store"], windows=[2])
    rolling_combined_train_part = rolling_combined[rolling_combined["Date"].isin(train_window["Date"])]

    merged = rolling_train_alone.merge(
        rolling_combined_train_part, on=["Store", "Date"], suffixes=("_alone", "_combined")
    )
    assert_series_equal(
        merged["rolling_mean_2w_alone"], merged["rolling_mean_2w_combined"],
        check_names=False,
        obj="rolling_mean_2w: train_window đơn lẻ vs. train_window trong bảng gộp",
    )


def test_valid_window_first_row_rolling_uses_real_train_history():
    """A46: dòng đầu valid_window phải có rolling_mean tính từ dữ liệu thật
    cuối train_window, không NaN oan do bị tính riêng lẻ."""
    full = _make_continuous_store_series(n_weeks=6)
    train_window = full.iloc[:4].copy()
    valid_window = full.iloc[4:].copy()

    valid_no_target = valid_window.drop(columns=["Weekly_Sales"])
    combined = pd.concat([train_window, valid_no_target], ignore_index=True, sort=False)
    rolling_combined = add_rolling_features(combined, group_cols=["Store"], windows=[2])

    first_valid_date = valid_window["Date"].min()
    first_valid_row = rolling_combined[rolling_combined["Date"] == first_valid_date].iloc[0]

    assert not pd.isna(first_valid_row["rolling_mean_2w"]), (
        "rolling_mean_2w của dòng đầu valid_window phải tính được từ 2 tuần "
        "cuối train_window thật, không NaN oan"
    )
