"""Data loaders cho stage 1: Data Ingestion & Validation."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from sales_forecast.ingestion.validators import DataContractError


def _read_csv_typed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(bool)
    return df


def load_raw_data(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Tải 4 file CSV gốc từ data/raw/, ép kiểu Date/IsHoliday theo schema.

    Returns:
        Dict với key: 'train', 'test', 'features', 'stores'
    """
    raw_dir = Path(raw_dir)
    return {
        "train": _read_csv_typed(raw_dir / "train.csv"),
        "test": _read_csv_typed(raw_dir / "test.csv"),
        "features": _read_csv_typed(raw_dir / "features.csv"),
        "stores": pd.read_csv(raw_dir / "stores.csv"),
    }


def aggregate_to_store_date(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gộp đơn vị dự báo từ (Store, Dept, Date) về (Store, Date) — bỏ Dept.

    Xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo: (Store, Dept,
    Date) -> (Store, Date)". Đây là thay đổi ĐƠN VỊ QUAN SÁT (không phải 1
    feature), nên phải chạy NGAY SAU validate schema raw, TRƯỚC Temporal Split
    — mọi giai đoạn sau (split, feature engineering, model, evaluation) đều
    thao tác trên grain (Store, Date) đã gộp, nhất quán với nguyên tắc "cắt
    mốc thời gian trước, tính feature sau".

    train_df: SUM Weekly_Sales theo (Store, Date) trên toàn bộ Dept có mặt
    tuần đó — giữ nguyên tinh thần "không xử lý Weekly_Sales âm" đã chốt
    trước đó, dòng âm lẻ tẻ vẫn được cộng vào tổng, không loại trừ.

    test_df: không có Weekly_Sales (target cần dự báo) — chỉ drop cột Dept và
    drop_duplicates theo (Store, Date).

    IsHoliday: lấy .first() sau khi đã assert nunique()==1 trong mỗi
    (Store, Date) — nếu dữ liệu vi phạm giả định này (không nhất quán giữa
    các Dept cùng Store-tuần), raise DataContractError thay vì âm thầm chọn
    sai giá trị.
    """

    def _assert_isholiday_consistent(df: pd.DataFrame, label: str) -> None:
        n_unique = df.groupby(["Store", "Date"])["IsHoliday"].nunique()
        bad = n_unique[n_unique > 1]
        if len(bad) > 0:
            raise DataContractError(
                f"IsHoliday không nhất quán giữa các Dept trong cùng (Store, Date) "
                f"ở '{label}' — {len(bad)} tổ hợp vi phạm. Không thể dùng .first() "
                f"an toàn khi aggregate_to_store_date. Ví dụ vi phạm:\n"
                f"{bad.head().to_string()}"
            )

    _assert_isholiday_consistent(train_df, "train")
    train_agg = (
        train_df.groupby(["Store", "Date"], as_index=False)
        .agg(Weekly_Sales=("Weekly_Sales", "sum"), IsHoliday=("IsHoliday", "first"))
    )

    _assert_isholiday_consistent(test_df, "test")
    test_agg = (
        test_df.drop(columns=["Dept"])
        .drop_duplicates(subset=["Store", "Date"])
        .reset_index(drop=True)
    )

    return train_agg, test_agg


def join_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join train/test với features.csv theo (Store, Date) — KHÔNG theo Dept.

    features.csv không có cột Dept (docs/03_data_io_diagram.md mục 1): nhiều
    Dept của cùng 1 Store dùng chung 1 dòng features, nên join phải giữ
    nguyên số dòng của `df` (không nhân bản, không mất dòng).

    IsHoliday tồn tại ở cả 2 nguồn — giữ lại cả hai dưới hậu tố _train/_features
    để phát hiện lệch nhau thay vì âm thầm lấy 1 nguồn (docs/03_data_io_diagram.md
    mục 1, giả định A4 ở docs/05_test_plan.md).
    """
    left = df.rename(columns={"IsHoliday": "IsHoliday_train"})
    right = features.rename(columns={"IsHoliday": "IsHoliday_features"})
    return left.merge(right, on=["Store", "Date"], how="left")
