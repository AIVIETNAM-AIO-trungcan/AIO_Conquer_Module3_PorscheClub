"""Test end-to-end luồng chính của pipelines/run_train_multistep.py (A56) —
docs/05_test_plan.md mục 1.

Không gọi trực tiếp _run() (đòi hỏi toàn bộ I/O: RunContext thật, file
configs/*.yaml, data/raw/ thật) — thay vào đó dựng fixture đủ lớn (nhiều
Store, nhiều tuần) để lắp đúng chuỗi build_feature_matrix -> split ->
add_horizon_targets -> _select_feature_cols -> fit/predict -> metric, xác
nhận không crash và output đúng cấu trúc, kể cả khi 1 horizon có rất ít
dòng valid sau dropna.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from sales_forecast.evaluation.metrics import wape, weighted_mae, weighted_mape
from sales_forecast.features.horizon_target import add_horizon_targets
from sales_forecast.features.pipeline import build_feature_matrix
from sales_forecast.ingestion.loaders import aggregate_to_store_date
from sales_forecast.models.direct_multihorizon import (
    make_direct_decision_tree,
    make_direct_random_forest,
)
from sales_forecast.splitting.ratio_split import split_by_date_ratio


def _make_raw(n_stores: int = 3, n_weeks: int = 70):
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="7D")
    rng = np.random.default_rng(42)
    rows = []
    for store in range(1, n_stores + 1):
        for date in dates:
            rows.append({
                "Store": store, "Dept": 1, "Date": date,
                "Weekly_Sales": 1000.0 + 10 * store + rng.normal(scale=20),
                "IsHoliday": False,
            })
    train_df = pd.DataFrame(rows)
    test_df = train_df[["Store", "Dept", "Date", "IsHoliday"]].iloc[:0]  # test rỗng, đủ dùng cho test này
    features_df = pd.DataFrame({
        "Store": [s for s in range(1, n_stores + 1) for _ in dates],
        "Date": list(dates) * n_stores,
        "Temperature": rng.normal(size=n_weeks * n_stores),
        "Fuel_Price": rng.normal(size=n_weeks * n_stores),
        "CPI": rng.normal(size=n_weeks * n_stores),
        "Unemployment": rng.normal(size=n_weeks * n_stores),
        "IsHoliday": [False] * (n_weeks * n_stores),
    })
    return train_df, test_df, features_df


def _select_feature_cols(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    drop_cols = {"Date", "Weekly_Sales"} | set(target_cols)
    return [c for c in df.columns if c not in drop_cols]


def test_pipeline_main_flow_runs_end_to_end_and_produces_metrics_table():
    """A56: chuỗi build_feature_matrix -> split -> horizon targets ->
    fit/predict -> metric chạy hết không crash, sinh bảng metric đủ cột cho
    2 model x N horizon."""
    train_df, test_df, features_df = _make_raw(n_stores=3, n_weeks=70)
    train_agg, test_agg = aggregate_to_store_date(train_df, test_df)
    fm = build_feature_matrix(train_agg, test_agg, features_df)
    fm = fm.dropna(subset=["lag_52w"])
    train_master = fm[fm["Weekly_Sales"].notna()]
    train_w, valid_w = split_by_date_ratio(train_master, ratio=2 / 3)

    horizon = 5  # nhỏ hơn 10 để fixture 70 tuần đủ dòng valid sau dropna
    train_w_t = add_horizon_targets(train_w, horizon=horizon)
    valid_w_t = add_horizon_targets(valid_w, horizon=horizon)
    target_cols = [f"target_t+{h}" for h in range(1, horizon + 1)]
    feature_cols = _select_feature_cols(train_w_t, target_cols)

    X_train = train_w_t[feature_cols]
    y_train_multi = train_w_t[target_cols]
    X_valid = valid_w_t[feature_cols]
    is_holiday_valid = valid_w_t["IsHoliday"].to_numpy()

    models = {
        "direct_decision_tree": make_direct_decision_tree(
            horizon=horizon, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42
        ),
        "direct_random_forest": make_direct_random_forest(
            horizon=horizon, n_estimators=5, max_depth=3, min_samples_split=2,
            min_samples_leaf=1, n_jobs=1, random_state=42,
        ),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(X_train, y_train_multi)
        for h in range(1, horizon + 1):
            target_col = f"target_t+{h}"
            mask = valid_w_t[target_col].notna()
            if mask.sum() == 0:
                continue
            y_true_h = valid_w_t.loc[mask, target_col].to_numpy()
            y_pred_h = model.predict_horizon(X_valid.loc[mask], h).to_numpy()
            is_holiday_h = is_holiday_valid[mask.to_numpy()]
            rows.append({
                "model": model_name, "horizon": h,
                "mae": mean_absolute_error(y_true_h, y_pred_h),
                "rmse": root_mean_squared_error(y_true_h, y_pred_h),
                "wape": wape(y_true_h, y_pred_h),
                "wmae": weighted_mae(y_true_h, y_pred_h, is_holiday_h),
                "wmape": weighted_mape(y_true_h, y_pred_h, is_holiday_h),
            })

    metrics_df = pd.DataFrame(rows)
    assert not metrics_df.empty
    assert set(metrics_df["model"].unique()) == {"direct_decision_tree", "direct_random_forest"}
    for col in ("mae", "rmse", "wape", "wmae", "wmape"):
        assert metrics_df[col].notna().all()
        assert np.isfinite(metrics_df[col]).all()


def test_pipeline_does_not_crash_when_horizon_has_very_few_valid_rows():
    """A56: horizon xa (gần hết dữ liệu valid do dropna target) không được
    crash pipeline — chỉ đơn giản là ít dòng đánh giá hơn, miễn train vẫn còn
    đủ dữ liệu để fit mọi horizon (n_weeks đủ lớn để tách biệt 2 tình huống:
    "valid ít dòng" ở đây, vs "train 0 dòng" ở test riêng bên dưới)."""
    train_df, test_df, features_df = _make_raw(n_stores=2, n_weeks=100)
    train_agg, test_agg = aggregate_to_store_date(train_df, test_df)
    fm = build_feature_matrix(train_agg, test_agg, features_df)
    fm = fm.dropna(subset=["lag_52w"])
    train_master = fm[fm["Weekly_Sales"].notna()]
    train_w, valid_w = split_by_date_ratio(train_master, ratio=2 / 3)

    horizon = 10  # đủ lớn để horizon xa có rất ít (có thể 0) dòng VALID còn lại
    train_w_t = add_horizon_targets(train_w, horizon=horizon)
    valid_w_t = add_horizon_targets(valid_w, horizon=horizon)
    target_cols = [f"target_t+{h}" for h in range(1, horizon + 1)]
    feature_cols = _select_feature_cols(train_w_t, target_cols)

    model = make_direct_decision_tree(
        horizon=horizon, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42
    )
    model.fit(train_w_t[feature_cols], train_w_t[target_cols])  # khong raise - train du du lieu

    X_valid = valid_w_t[feature_cols]
    n_evaluated = 0
    for h in range(1, horizon + 1):
        mask = valid_w_t[f"target_t+{h}"].notna()
        if mask.sum() == 0:
            continue  # đúng hành vi mong đợi: bỏ qua, không crash
        y_pred_h = model.predict_horizon(X_valid.loc[mask], h)
        assert len(y_pred_h) == mask.sum()
        n_evaluated += 1
    assert n_evaluated >= 1  # ít nhất horizon gần phải còn dữ liệu để đánh giá


def test_fit_raises_clear_error_when_train_has_zero_rows_for_a_horizon():
    """A56: neu du lieu TRAIN qua ngan so voi horizon (0 dong kha dung cho 1
    horizon xa sau dropna), fit() phai raise ValueError ro rang thay vi de
    sklearn nem loi kho hieu o tang duoi - phat hien that khi test voi
    fixture nho (65 tuan, horizon=10)."""
    train_df, test_df, features_df = _make_raw(n_stores=2, n_weeks=65)
    train_agg, test_agg = aggregate_to_store_date(train_df, test_df)
    fm = build_feature_matrix(train_agg, test_agg, features_df)
    fm = fm.dropna(subset=["lag_52w"])
    train_master = fm[fm["Weekly_Sales"].notna()]
    train_w, _ = split_by_date_ratio(train_master, ratio=2 / 3)

    horizon = 10
    train_w_t = add_horizon_targets(train_w, horizon=horizon)
    target_cols = [f"target_t+{h}" for h in range(1, horizon + 1)]
    feature_cols = _select_feature_cols(train_w_t, target_cols)

    model = make_direct_decision_tree(
        horizon=horizon, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42
    )
    try:
        model.fit(train_w_t[feature_cols], train_w_t[target_cols])
        assert False, "Phai raise ValueError khi 1 horizon co 0 dong train"
    except ValueError as e:
        assert "horizon" in str(e).lower()
