"""Data Contract cho stage 1: Data Ingestion & Validation.

Đây là nơi DUY NHẤT kiểm tra schema/toàn vẹn dữ liệu của pipeline
(CLAUDE.md mục 4 rule 10 tương ứng — "Data Contract là nơi duy nhất
kiểm tra schema", docs/02_pipeline_architecture.md mục 2).
"""

import pandas as pd
import pandera.errors

from sales_forecast.ingestion.schema import (
    features_schema,
    stores_schema,
    test_aggregated_schema,
    test_schema,
    train_aggregated_schema,
    train_schema,
)


class DataContractError(Exception):
    """Raised khi dữ liệu không khớp Data Contract — dừng pipeline, không âm thầm bỏ qua."""


def _validate(df: pd.DataFrame, schema, label: str) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except pandera.errors.SchemaErrors as exc:
        raise DataContractError(f"Data Contract vi phạm cho '{label}':\n{exc.failure_cases}") from exc


def validate_train_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Xác thực schema train.csv. Weekly_Sales âm được PHÉP (đã chốt ở docs/00_decisions.md)."""
    return _validate(df, train_schema, "train")


def validate_test_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Xác thực schema test.csv (không có cột Weekly_Sales — đây là target cần dự báo)."""
    return _validate(df, test_schema, "test")


def validate_features_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Xác thực schema features.csv. MarkDown NaN là hợp lệ (MNAR, xem docs/00_decisions.md)."""
    return _validate(df, features_schema, "features")


def validate_stores_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Xác thực schema stores.csv."""
    return _validate(df, stores_schema, "stores")


def validate_train_aggregated_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Xác thực schema train SAU aggregate_to_store_date (không còn Dept).
    Xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"."""
    return _validate(df, train_aggregated_schema, "train_aggregated")


def validate_test_aggregated_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Xác thực schema test SAU aggregate_to_store_date (không còn Dept)."""
    return _validate(df, test_aggregated_schema, "test_aggregated")
