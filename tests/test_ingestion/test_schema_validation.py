"""Test Data Contract (A1, A14) — docs/05_test_plan.md mục 2.2."""

import pytest

from sales_forecast.ingestion.loaders import aggregate_to_store_date
from sales_forecast.ingestion.validators import (
    DataContractError,
    validate_test_aggregated_schema,
    validate_train_aggregated_schema,
    validate_train_schema,
)


def test_train_schema_valid(sample_train):
    """Giả định A1: schema đúng thì validate không raise."""
    validate_train_schema(sample_train)  # không raise


def test_train_schema_rejects_missing_column(sample_train):
    """Giả định A1: thiếu cột bắt buộc phải raise, không âm thầm bỏ qua."""
    broken = sample_train.drop(columns=["IsHoliday"])
    with pytest.raises(DataContractError):
        validate_train_schema(broken)


def test_negative_sales_preserved(sample_train):
    """Giả định A14: Weekly_Sales âm KHÔNG bị clip về 0 khi ingest.

    Theo docs/00_decisions.md, sales âm được giữ nguyên (tín hiệu thật:
    trả hàng/điều chỉnh sổ sách), Data Contract phải CHO PHÉP giá trị âm.
    """
    assert (sample_train["Weekly_Sales"] < 0).any(), \
        "Fixture phải giữ ít nhất 1 giá trị âm để test có ý nghĩa"
    validated = validate_train_schema(sample_train)
    assert (validated["Weekly_Sales"] < 0).any(), \
        "Data Contract không được clip/loại bỏ Weekly_Sales âm"


def test_train_aggregated_schema_valid_after_aggregate(sample_train, sample_test):
    """Schema aggregated (Store, Date, không còn Dept) phải validate thành công
    trên output thật của aggregate_to_store_date — xem docs/00_decisions.md
    [2026-08-19] "Đổi đơn vị dự báo"."""
    train_agg, test_agg = aggregate_to_store_date(sample_train, sample_test)
    validate_train_aggregated_schema(train_agg)  # không raise
    validate_test_aggregated_schema(test_agg)  # không raise


def test_train_aggregated_schema_does_not_require_dept(sample_train_aggregated):
    """Schema aggregated KHÔNG được yêu cầu cột Dept — dữ liệu chỉ có
    (Store, Date, Weekly_Sales, IsHoliday) vẫn phải validate thành công."""
    assert "Dept" not in sample_train_aggregated.columns
    validate_train_aggregated_schema(sample_train_aggregated)  # không raise
