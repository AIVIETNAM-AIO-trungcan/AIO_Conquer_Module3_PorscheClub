"""Sinh target theo horizon cho chiến lược Direct multi-step.

Khớp notebooks/viet/multi_step/direct_way/direct_multimodel_DTree.ipynb và
direct_multimodel_rf.ipynb (Cell 'cell-targets'): mỗi horizon h có 1 cột
target_t+{h} riêng, dùng để fit 1 model độc lập cho mỗi h (xem
src/sales_forecast/models/direct_multihorizon.py).

KHÔNG đăng ký vào configs/features.yaml/ALL_BLOCKS/build_feature_matrix() —
đây không phải feature block bật/tắt của Giai đoạn 3 (sinh target, không sinh
feature X), phải chạy SAU khi đã có feature_matrix, gọi trực tiếp từ pipeline
orchestration đa bước (pipelines/run_train_multistep.py).
"""

import pandas as pd


def add_horizon_targets(
    df: pd.DataFrame,
    group_col: str = "Store",
    target_col: str = "Weekly_Sales",
    horizon: int = 10,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Sinh target_t+{h} cho h=1..horizon bằng group(...)[target_col].shift(-h).

    Tự sort theo (group_col, date_col) trước khi shift (không giả định caller
    đã sort). Trả DataFrame mới, KHÔNG sửa df gốc. Dòng cuối mỗi group không
    đủ h tuần dữ liệu tương lai -> NaN (đúng ngữ nghĩa "chưa biết", không leak
    — shift(-h) chỉ dịch trong cùng group, không lấy nhầm dòng của group khác).
    """
    out = df.sort_values([group_col, date_col]).reset_index(drop=True).copy()
    for h in range(1, horizon + 1):
        out[f"target_t+{h}"] = out.groupby(group_col)[target_col].shift(-h)
    return out
