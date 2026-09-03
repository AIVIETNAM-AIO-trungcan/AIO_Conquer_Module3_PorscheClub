"""Metrics dùng chung cho Evaluation Layer.

Theo docs/00_decisions.md: báo cáo CẢ WMAE và WMAPE, cùng trọng số IsHoliday
x5 (đúng luật đánh giá gốc Kaggle) — không chọn một làm "chính" duy nhất.
"""

import numpy as np


def _weights(is_holiday: np.ndarray, holiday_weight: float) -> np.ndarray:
    return np.where(np.asarray(is_holiday, dtype=bool), holiday_weight, 1.0)


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
    holiday_weight: float = 5.0,
) -> float:
    """Weighted MAE — tuần lễ (IsHoliday=True) có trọng số gấp `holiday_weight` lần."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = _weights(is_holiday, holiday_weight)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def weighted_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
    holiday_weight: float = 5.0,
    epsilon: float = 1e-8,
) -> float:
    """Weighted MAPE — cùng trọng số IsHoliday x5. `epsilon` tránh chia cho 0
    khi Weekly_Sales = 0 (73 dòng trong train, xem docs/01_ideation.md mục 2.1).

    GIỚI HẠN ĐÃ BIẾT: dataset có hàng trăm dòng Weekly_Sales gần 0 nhưng khác 0
    (vd. 0.01-0.2). Với các dòng này, APE = |y_true-y_pred|/|y_true| có thể ra
    giá trị cực lớn dù sai số tuyệt đối rất nhỏ, khiến WMAPE tổng bị một số ít
    dòng chi phối. Theo docs/00_decisions.md, KHÔNG loại trừ các dòng này (giữ
    nguyên tín hiệu thật) — WMAE vẫn là metric ổn định để đối chiếu, WMAPE cần
    diễn giải cùng WMAE, không dùng riêng lẻ. Phân tích nhóm sales gần 0 thuộc
    Giai đoạn 7 (Evaluation & Error Analysis)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = _weights(is_holiday, holiday_weight)
    ape = np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)
    return float(np.sum(w * ape) / np.sum(w))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE KHÔNG trọng số IsHoliday — dùng để đối chiếu trực tiếp với số
    liệu notebooks/viet/multi_step/direct_way/*.ipynb (dự báo Direct
    multi-step) đã báo cáo.

    CHÚ Ý: mẫu số PHẢI có abs (np.abs(y_true).sum()) — KHÔNG lặp lỗi thiếu abs
    đã phát hiện ở 1 cell của direct_multimodel_rf.ipynb
    (wape = |y-pred|.sum()/y_val.sum(), thiếu abs ở mẫu số, khác cell khác
    cùng file có abs đúng). Với Weekly_Sales âm (giữ nguyên theo quyết định
    đã chốt — xem docs/00_decisions.md "Xử lý Weekly_Sales âm"), thiếu abs ở
    mẫu số có thể cho WAPE âm hoặc vô nghĩa khi tổng y_true âm/gần 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())
