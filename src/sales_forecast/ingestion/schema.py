"""Schema validation cho stage 1: Data Ingestion & Validation.

Data Contract duy nhất của pipeline (CLAUDE.md mục 4 rule — kiểm tra schema
chỉ ở đây, các giai đoạn sau không lặp lại kiểm tra rải rác).

Đơn vị dự báo đã đổi sang (Store, Date) — xem docs/00_decisions.md
[2026-08-19] "Đổi đơn vị dự báo: (Store, Dept, Date) -> (Store, Date)".
`train_schema`/`test_schema` bên dưới validate dữ liệu RAW (train.csv/test.csv
gốc, vẫn có cột Dept thật) NGAY SAU khi đọc file — đây vẫn là hợp đồng đúng
với chính file CSV gốc. Bước aggregate_to_store_date() (ingestion/loaders.py)
chạy SAU bước validate raw này, TRƯỚC Temporal Split — output của nó dùng
`train_aggregated_schema`/`test_aggregated_schema` (không còn Dept) làm hợp
đồng cho mọi giai đoạn từ đó trở đi.
"""

import pandera as pa

train_schema = pa.DataFrameSchema({
    "Store": pa.Column(int),
    "Dept": pa.Column(int),
    "Date": pa.Column("datetime64[ns]"),
    "Weekly_Sales": pa.Column(float, nullable=False),
    "IsHoliday": pa.Column(bool),
})

test_schema = pa.DataFrameSchema({
    "Store": pa.Column(int),
    "Dept": pa.Column(int),
    "Date": pa.Column("datetime64[ns]"),
    "IsHoliday": pa.Column(bool),
})

train_aggregated_schema = pa.DataFrameSchema({
    "Store": pa.Column(int),
    "Date": pa.Column("datetime64[ns]"),
    "Weekly_Sales": pa.Column(float, nullable=False),
    "IsHoliday": pa.Column(bool),
})

test_aggregated_schema = pa.DataFrameSchema({
    "Store": pa.Column(int),
    "Date": pa.Column("datetime64[ns]"),
    "IsHoliday": pa.Column(bool),
})

features_schema = pa.DataFrameSchema({
    "Store": pa.Column(int),
    "Date": pa.Column("datetime64[ns]"),
    "Temperature": pa.Column(float),
    "Fuel_Price": pa.Column(float),
    "CPI": pa.Column(float, nullable=True),
    "Unemployment": pa.Column(float, nullable=True),
    "IsHoliday": pa.Column(bool),
    "MarkDown1": pa.Column(float, nullable=True, required=False),
    "MarkDown2": pa.Column(float, nullable=True, required=False),
    "MarkDown3": pa.Column(float, nullable=True, required=False),
    "MarkDown4": pa.Column(float, nullable=True, required=False),
    "MarkDown5": pa.Column(float, nullable=True, required=False),
})

stores_schema = pa.DataFrameSchema({
    "Store": pa.Column(int),
    "Type": pa.Column(str),
    "Size": pa.Column(int),
})
