"""Test độc lập feature block (A9) — docs/05_test_plan.md mục 2.5."""

from sales_forecast.features.pipeline import build_feature_matrix


def test_disabling_markdown_block_does_not_break_pipeline(
    sample_train_aggregated, sample_test_aggregated, sample_features
):
    """Giả định A9: tắt block MarkDown qua config không được làm hỏng các block khác
    (Lag/Calendar/Encoding vẫn phải chạy đúng)."""
    fm_full = build_feature_matrix(
        sample_train_aggregated, sample_test_aggregated, sample_features,
        enabled_blocks=["lag_rolling", "calendar", "markdown", "encoding"],
    )
    fm_no_markdown = build_feature_matrix(
        sample_train_aggregated, sample_test_aggregated, sample_features,
        enabled_blocks=["lag_rolling", "calendar", "encoding"],
    )
    assert "MarkDown1" not in fm_no_markdown.columns
    assert "lag_1w" in fm_no_markdown.columns
    assert set(fm_no_markdown["Store"]) == set(fm_full["Store"])
