"""Data loaders cho stage 1: Data Ingestion & Validation."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


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
    Date) -> (Store, Date)" và mục quyết định mới nhất "Đồng bộ xử lý dữ liệu
    theo notebooks/01. Preprocessing.ipynb" (GHI ĐÈ phần IsHoliday của quyết
    định trên). Đây là thay đổi ĐƠN VỊ QUAN SÁT (không phải 1 feature), nên
    phải chạy NGAY SAU validate schema raw, TRƯỚC Temporal Split.

    train_df: SUM Weekly_Sales theo (Store, Date, IsHoliday) trên toàn bộ Dept
    có mặt tuần đó — giữ nguyên tinh thần "không xử lý Weekly_Sales âm" đã
    chốt trước đó, dòng âm lẻ tẻ vẫn được cộng vào tổng, không loại trừ.

    test_df: không có Weekly_Sales (target cần dự báo) — chỉ drop cột Dept và
    drop_duplicates theo (Store, Date, IsHoliday).

    IsHoliday: group theo cả IsHoliday (khớp notebooks/01. Preprocessing.ipynb
    Cell 6) — nếu 2 Dept cùng (Store, Date) có IsHoliday khác nhau (lệch dữ
    liệu hiếm gặp), kết quả sẽ TÁCH THÀNH NHIỀU DÒNG riêng theo từng giá trị
    IsHoliday quan sát được, KHÔNG raise lỗi (đảo ngược quyết định cũ dùng
    .first() + assert nhất quán).
    """
    train_agg = (
        train_df.groupby(["Store", "Date", "IsHoliday"], as_index=False)
        .agg(Weekly_Sales=("Weekly_Sales", "sum"))
    )

    test_agg = (
        test_df.drop(columns=["Dept"])
        .drop_duplicates(subset=["Store", "Date", "IsHoliday"])
        .reset_index(drop=True)
    )

    return train_agg, test_agg


def join_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join train/test với features.csv theo (Store, Date, IsHoliday).

    Quyết định team (docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo
    notebooks/01. Preprocessing.ipynb" — GHI ĐÈ quyết định join chỉ theo
    (Store, Date) trước đó): gộp IsHoliday vào khóa join thay vì giữ 2 nguồn
    IsHoliday riêng để so sánh. Nếu IsHoliday giữa `df` và `features` lệch
    nhau ở cùng (Store, Date), dòng đó sẽ KHÔNG khớp trên khóa join — các cột
    đến từ `features` sẽ là NaN cho dòng đó, thay vì báo lỗi tường minh.
    """
    return df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
