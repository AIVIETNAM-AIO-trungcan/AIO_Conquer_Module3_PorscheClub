"""Test MarkDown flag (A7) — docs/05_test_plan.md mục 2.4."""

from sales_forecast.features.markdown_promo import add_markdown_features


def test_markdown_nan_becomes_explicit_flag(sample_features):
    """Giả định A7: NaN ở MarkDown1 phải tạo has_markdown_1=False,
    KHÔNG được lặng lẽ fillna(0) khiến 'không có dữ liệu' trông giống 'khuyến mãi = 0đ'."""
    df = add_markdown_features(sample_features)
    nan_rows = df[df["MarkDown1"].isna()]
    assert (nan_rows["has_markdown_1"] == False).all()
