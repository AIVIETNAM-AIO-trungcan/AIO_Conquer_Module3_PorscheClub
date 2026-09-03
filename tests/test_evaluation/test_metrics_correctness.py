"""Test metrics (A11) — docs/05_test_plan.md mục 2.6, mở rộng WMAPE theo docs/00_decisions.md."""

import numpy as np

from sales_forecast.evaluation.metrics import wape, weighted_mae, weighted_mape


def test_weighted_mae_holiday_weight_is_5x():
    """Giả định A11: tuần lễ (IsHoliday=True) phải có trọng số gấp 5 lần
    tuần thường trong công thức WMAE, đúng luật đánh giá gốc Kaggle.

    Dùng sai số KHÁC nhau giữa 2 dòng (không phải cùng =10) để trọng số
    thực sự ảnh hưởng tới kết quả cuối — nếu mọi residual bằng nhau thì
    weighted average trùng plain average bất kể trọng số, không phải test
    có ý nghĩa cho việc trọng số hoạt động đúng.
    """
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([80.0, 90.0])   # sai số 20 (non-holiday) và 10 (holiday)
    is_holiday = np.array([False, True])
    wmae = weighted_mae(y_true, y_pred, is_holiday)
    plain_mae = np.mean(np.abs(y_true - y_pred))
    assert wmae != plain_mae
    expected = (20 * 1 + 10 * 5) / (1 + 5)
    assert np.isclose(wmae, expected)


def test_weighted_mape_holiday_weight_is_5x():
    """Mở rộng A11 (docs/00_decisions.md): WMAPE cũng phải dùng trọng số
    IsHoliday x5, báo cáo song song với WMAE thay vì chỉ chọn một metric."""
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 180.0])  # APE = 0.10 ở cả 2 dòng
    is_holiday = np.array([False, True])
    wmape = weighted_mape(y_true, y_pred, is_holiday)
    expected = (0.10 * 1 + 0.10 * 5) / (1 + 5)
    assert np.isclose(wmape, expected, atol=1e-6)


def test_weighted_mape_handles_zero_sales_without_inf():
    """Weekly_Sales có thể bằng 0 (73 dòng trong train thật) — WMAPE không
    được trả về inf/NaN khi y_true = 0."""
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 100.0])
    is_holiday = np.array([False, False])
    wmape = weighted_mape(y_true, y_pred, is_holiday)
    assert np.isfinite(wmape)


def test_wape_correct_formula_with_abs_in_denominator():
    """A55: WAPE phải dùng đúng công thức có abs() ở MẪU SỐ. Dựng y_true có
    số âm để phân biệt rõ kết quả đúng (có abs) vs sai (thiếu abs, như 1 cell
    của direct_multimodel_rf.ipynb) — 2 công thức cho ra kết quả khác nhau
    rõ rệt khi tổng y_true có dấu âm."""
    y_true = np.array([-100.0, 50.0])  # sum = -50, abs(sum từng phần tử) = 150
    y_pred = np.array([-90.0, 60.0])   # |errors| = [10, 10], sum = 20
    result = wape(y_true, y_pred)
    correct_with_abs = 20.0 / 150.0  # đúng: abs(y_true).sum() = 150
    wrong_without_abs = 20.0 / -50.0  # sai (lỗi notebook): y_true.sum() = -50 (âm)
    assert np.isclose(result, correct_with_abs)
    assert not np.isclose(result, wrong_without_abs)


def test_wape_basic_positive_case():
    """A55: trường hợp cơ bản không có số âm, WAPE = sum(|error|)/sum(|y_true|)."""
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 180.0])
    result = wape(y_true, y_pred)
    expected = (10.0 + 20.0) / (100.0 + 200.0)
    assert np.isclose(result, expected)
