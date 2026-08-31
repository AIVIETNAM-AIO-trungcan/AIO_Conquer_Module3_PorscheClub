"""Test MarkDown fillna(0) (A7 DEPRECATED, A48) — docs/05_test_plan.md mục 2.4.

A7 gốc (flag has_markdown_{i}, giữ NaN) không còn áp dụng — xem
docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo notebooks/01. Preprocessing.ipynb".
"""

from sales_forecast.features.markdown_promo import add_markdown_features


def test_markdown_nan_becomes_zero(sample_features):
    """A48: NaN ở MarkDown1 phải trở thành 0 (fillna(0)), không còn cột
    has_markdown_1 nào được tạo ra."""
    df = add_markdown_features(sample_features)
    assert not df["MarkDown1"].isna().any()
    nan_original_rows = sample_features[sample_features["MarkDown1"].isna()]
    filled_rows = df.loc[nan_original_rows.index]
    assert (filled_rows["MarkDown1"] == 0).all()
    assert "has_markdown_1" not in df.columns
