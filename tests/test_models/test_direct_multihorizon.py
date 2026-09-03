"""Test DirectMultiHorizonModel (A54) — docs/05_test_plan.md mục 1.

Khớp notebooks/viet/multi_step/direct_way/*.ipynb: mỗi horizon h có 1
estimator độc lập, fit riêng, predict riêng.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from sales_forecast.models.direct_multihorizon import (
    DirectMultiHorizonModel,
    make_direct_decision_tree,
    make_direct_random_forest,
)


def _make_xy(n_rows: int = 40, horizon: int = 3):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Store": pd.Categorical([1] * n_rows),
        "feat_a": rng.normal(size=n_rows),
        "feat_b": rng.normal(size=n_rows),
    })
    y_multi = pd.DataFrame({
        f"target_t+{h}": rng.normal(size=n_rows) for h in range(1, horizon + 1)
    })
    # Mô phỏng dòng cuối chuỗi không đủ target tương lai -> NaN
    y_multi.loc[n_rows - 1, "target_t+3"] = np.nan
    return X, y_multi


def test_fit_creates_independent_estimator_per_horizon():
    """A54: mỗi horizon có 1 estimator riêng (object khác nhau), không dùng
    chung 1 model cho mọi horizon."""
    X, y_multi = _make_xy(horizon=3)
    model = DirectMultiHorizonModel(
        base_estimator_factory=lambda: DecisionTreeRegressor(max_depth=3, random_state=42),
        horizon=3,
    )
    model.fit(X, y_multi)
    assert len(model._models) == 3
    assert model._models[1] is not model._models[2]
    assert model._models[2] is not model._models[3]


def test_fit_drops_nan_target_per_horizon_independently():
    """A54: dropna theo TỪNG horizon riêng — horizon có ít dữ liệu hơn (do
    NaN ở cuối chuỗi) vẫn fit được, không ảnh hưởng horizon khác."""
    X, y_multi = _make_xy(horizon=3)
    model = DirectMultiHorizonModel(
        base_estimator_factory=lambda: DecisionTreeRegressor(max_depth=2, random_state=42),
        horizon=3,
    )
    model.fit(X, y_multi)  # không raise dù target_t+3 có 1 dòng NaN
    assert 3 in model._models


def test_predict_returns_dataframe_with_correct_shape_and_columns():
    """A54: predict() trả DataFrame index=X.index, đúng cột target_t+1..N."""
    X, y_multi = _make_xy(horizon=3)
    model = make_direct_decision_tree(
        horizon=3, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42
    )
    model.fit(X, y_multi)
    preds = model.predict(X)
    assert preds.shape == (len(X), 3)
    assert list(preds.columns) == ["target_t+1", "target_t+2", "target_t+3"]
    assert list(preds.index) == list(X.index)


def test_predict_horizon_matches_column_in_full_predict():
    """A54: predict_horizon(X, h) phải khớp đúng cột target_t+{h} trong
    kết quả của predict(X) đầy đủ."""
    X, y_multi = _make_xy(horizon=3)
    model = make_direct_random_forest(
        horizon=3, n_estimators=10, max_depth=3, min_samples_split=2,
        min_samples_leaf=1, n_jobs=1, random_state=42,
    )
    model.fit(X, y_multi)
    full = model.predict(X)
    single = model.predict_horizon(X, h=2)
    pd.testing.assert_series_equal(single, full["target_t+2"], check_names=False)


def test_predict_before_fit_raises():
    """A54: gọi predict() trước fit() phải raise lỗi rõ ràng, không crash mơ hồ."""
    model = make_direct_decision_tree(
        horizon=2, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42
    )
    X, _ = _make_xy(horizon=2)
    try:
        model.predict(X)
        assert False, "Phải raise RuntimeError khi chưa fit"
    except RuntimeError:
        pass


def test_predict_horizon_unknown_h_raises_key_error():
    """A54: predict_horizon với h chưa fit phải raise KeyError rõ ràng."""
    X, y_multi = _make_xy(horizon=2)
    model = make_direct_decision_tree(
        horizon=2, max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=42
    )
    model.fit(X, y_multi)
    try:
        model.predict_horizon(X, h=5)
        assert False, "Phải raise KeyError khi horizon chưa fit"
    except KeyError:
        pass
