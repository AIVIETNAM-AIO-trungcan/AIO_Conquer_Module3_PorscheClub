"""Test interface thống nhất (A12) — docs/05_test_plan.md mục 2.7.

Giới hạn ở 2 baseline model đã có; DT/RF/AdaBoost/GradientBoost/LightGBM/XGBoost
thật sẽ mở rộng bảng parametrize này khi được implement ở các bước sau.
"""

import pytest

from sales_forecast.features.store_encoding import encode_store
from sales_forecast.models.baseline import NaiveSameWeekLastYear, SimpleDecisionTreeBaseline

MODELS = [NaiveSameWeekLastYear, SimpleDecisionTreeBaseline]


@pytest.mark.parametrize("model_cls", MODELS)
def test_model_has_unified_interface(model_cls):
    """Giả định A12: mọi model, kể cả baseline, phải expose đúng .fit(X, y) / .predict(X)
    để Evaluation Layer dùng chung 1 đoạn code, không if/else riêng theo loại model."""
    model = model_cls()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


@pytest.mark.parametrize("model_cls", MODELS)
def test_model_fit_predict_roundtrip(model_cls, sample_train_aggregated):
    """fit rồi predict trên chính dữ liệu train phải trả về Series cùng độ dài,
    không lỗi, không toàn NaN. Dùng fixture aggregated (Store, Date) đúng
    granularity thật của pipeline (docs/00_decisions.md [2026-08-19])."""
    X = sample_train_aggregated.drop(columns=["Weekly_Sales"])
    y = sample_train_aggregated["Weekly_Sales"]
    model = model_cls().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert preds.notna().all()


def test_decision_tree_baseline_uses_store(sample_train_aggregated):
    """Giả định A24 (cập nhật, chỉ còn Store): sau block store_encoding
    (Store -> category), SimpleDecisionTreeBaseline không được âm thầm loại
    bỏ cột này qua select_dtypes — nếu không, DT baseline mất khả năng phân
    biệt cửa hàng, xem docs/progress_week1_report.md mục 3.2."""
    df = encode_store(sample_train_aggregated)
    X = df.drop(columns=["Weekly_Sales"])
    y = df["Weekly_Sales"]
    model = SimpleDecisionTreeBaseline().fit(X, y)
    assert "Store" in model._feature_cols
