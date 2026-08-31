# Test Plan — Test case cho từng giả định kiến trúc

**AIO Conquer 2026 — Module 03 · Project 3.2**

> Nguyên tắc: **mọi giả định quan trọng nêu trong `01_ideation.md`, `02_pipeline_architecture.md`, `03_data_io_diagram.md` đều phải có ít nhất 1 test case tương ứng.** Không có giả định "ngầm hiểu, tin tưởng bằng mắt". Test được viết bằng `pytest`, dữ liệu giả lập nhỏ (fixture), không phụ thuộc `data/raw/` thật để chạy nhanh và không rò rỉ dữ liệu thật vào CI.

---

## 1. Bảng ánh xạ Giả định → Test case

| # | Giả định (từ tài liệu ideation/kiến trúc) | Test case | File |
|---|---|---|---|
| A1 | Schema 4 file CSV đúng như khảo sát (cột, kiểu, khoảng giá trị) | `test_schema_validation.py` | `tests/test_ingestion/` |
| A2 | Ở bước ingestion (TRƯỚC aggregate), không phải mọi (Store, Dept) tồn tại — không dùng cross-join giả định lưới đầy đủ 45×81. Sau aggregate (2026-08-19), đơn vị là (Store, Date), không còn Dept | `test_join_integrity.py::test_no_cartesian_assumption` | `tests/test_ingestion/` |
| A3 | `features.csv` join theo (Store, Date), không theo Dept — join không được nhân bản/mất dòng. Sau aggregate (2026-08-19), cả target và exogenous features đều cùng grain (Store, Date), join trở thành 1-1 thay vì N-1 | `test_join_integrity.py::test_features_join_row_count`, `test_join_integrity.py::test_features_join_after_aggregate_is_one_to_one` | `tests/test_ingestion/` |
| A4 | **DEPRECATED (2026-08-31)** — `IsHoliday` khớp nhau giữa train/test và features khi join, lệch phải phát hiện qua 2 cột riêng. GHI ĐÈ: join nay theo (Store,Date,IsHoliday), lệch tạo NaN thay vì báo lỗi — xem A47 | `test_join_integrity.py::test_isholiday_mismatch_yields_nan_features_not_error` | `tests/test_ingestion/` |
| A5 | Temporal split không rò rỉ tương lai vào train/valid | `test_temporal_split_no_leakage.py` | `tests/test_splitting/` |
| A6 | Lag/rolling feature tại thời điểm t chỉ dùng dữ liệu ≤ t−1 | `test_lag_rolling_no_future_leak.py` | `tests/test_features/` |
| A7 | **DEPRECATED (2026-08-31)** — MarkDown NaN được hiểu là "không có khuyến mãi" qua flag `has_markdown`, không fillna(0) mù quáng. GHI ĐÈ: nay fillna(0) trực tiếp, không còn flag — xem A48 | `test_markdown_flag.py::test_markdown_nan_becomes_zero` | `tests/test_features/` |
| A8 | **DEPRECATED (2026-08-19)** — Cold-start (11 cặp Store-Dept chỉ có ở test) được nhận diện và xử lý riêng, không NaN âm thầm. Không còn áp dụng: đơn vị dự báo đã đổi sang (Store, Date), đã xác nhận KHÔNG có Store nào cold-start (cả 45 Store đều có trong train) — xem A43 | `test_cold_start_handling.py` | `tests/test_features/` |
| A9 | Mỗi feature block hoạt động độc lập, bật/tắt không phá vỡ pipeline | `test_feature_block_independence.py` | `tests/test_features/` |
| A10 | TimeSeriesSplit/walk-forward luôn giữ thứ tự thời gian (train luôn trước valid) | `test_cv_splitter_order.py` | `tests/test_evaluation/` |
| A11 | WMAE/WMAPE tính đúng công thức, trọng số IsHoliday đúng x5 | `test_metrics_correctness.py` | `tests/test_evaluation/` |
| A12 | Mọi model (baseline → LightGBM → XGBoost) tuân thủ cùng interface `.fit(X,y)/.predict(X)` | `test_model_interface_consistency.py` | `tests/test_models/` |
| A13 | Optuna không được thấy `test_window` trong quá trình tối ưu | `test_optuna_no_test_leakage.py` | `tests/test_models/` |
| A14 | Weekly_Sales âm không bị pipeline âm thầm clip về 0 ở bước load | `test_schema_validation.py::test_negative_sales_preserved` | `tests/test_ingestion/` |
| A15 | SHAP rank importance được so sánh ổn định qua các fold (không chỉ tính 1 lần) | `test_shap_stability.py` | `tests/test_explainability/` |
| A16 | `calib_window` (conformal) không trùng với `train_window` dùng để fit model | `test_conformal_prediction.py::test_conformal_no_train_calib_overlap` | `tests/test_evaluation/` |
| A17 | `calib_window` nằm sau `train_window` về thời gian (không random) | `test_conformal_prediction.py::test_conformal_calib_after_train` | `tests/test_evaluation/` |
| A18 | Quantile conformal dùng hiệu chỉnh hữu hạn mẫu `(n+1)(1-α)/n`, không phải `np.quantile` trần | `test_conformal_prediction.py::test_conformal_finite_sample_correction` | `tests/test_evaluation/` |
| A19 | Khoảng tin cậy luôn thỏa `y_lo ≤ y_pred ≤ y_hi` cho mọi dòng | `test_conformal_prediction.py::test_conformal_interval_contains_point_forecast` | `tests/test_evaluation/` |
| A20 | Empirical coverage trên `valid_window` nằm trong khoảng chấp nhận được quanh 95% | `test_conformal_prediction.py::test_conformal_empirical_coverage_reasonable` | `tests/test_evaluation/` |
| A21 | Dashboard báo lỗi rõ ràng (không crash traceback thô) khi thiếu file kết quả | `test_data_loader.py::test_data_loader_missing_file_friendly_error` | `tests/test_app/` |
| A22 | Bảng màu model trong dashboard nhất quán giữa các lần gọi | `test_theme.py::test_theme_color_mapping_deterministic` | `tests/test_app/` |
| A23 | Dashboard không tự train model (chỉ đọc file kết quả, không gọi `.fit()`) | `test_dashboard_has_no_training_calls` | `tests/test_app/` |
| A24 | `SimpleDecisionTreeBaseline` phải dùng được cột category Store (sau block `store_encoding`, đổi tên từ `store_dept_encoding` — 2026-08-19), không bị `select_dtypes` âm thầm loại bỏ (xem `progress_week1_report.md` mục 3.2) | `test_model_interface_consistency.py::test_decision_tree_baseline_uses_store` | `tests/test_models/` |
| A25 | 2 lần gọi `start_run()` liên tiếp (kể cả cùng giây, cùng pipeline_name) luôn sinh `run_id` khác nhau (suffix `_2`/`_3`...) | `test_run_tracking.py::test_run_id_never_collides_same_second` | `tests/test_utils/` |
| A26 | `start_run()`/ghi file qua `RunContext` không bao giờ ghi đè thư mục run đã tồn tại của lần chạy trước | `test_run_tracking.py::test_run_context_never_overwrites_previous_run` | `tests/test_utils/` |
| A27 | Sau `finalize(status="success")`, `get_latest_run_id()` trả đúng run vừa hoàn tất | `test_run_tracking.py::test_latest_pointer_matches_most_recent_successful_run` | `tests/test_utils/` |
| A28 | `finalize(status="failed")` (hoặc không gọi finalize) KHÔNG cập nhật `latest_run.txt` | `test_run_tracking.py::test_failed_run_does_not_update_latest_pointer` | `tests/test_utils/` |
| A29 | `reports/latest_run.txt` và `data/predictions/latest_run.txt` độc lập — 1 run chỉ ghi vào 1 base_dir không đụng pointer của base_dir kia | `test_run_tracking.py::test_latest_pointers_independent_per_base_dir` | `tests/test_utils/` |
| A30 | `manifest.json` chứa đủ trường bắt buộc (`run_id`, `pipeline_name`, `started_at`, `finished_at`, `status`, config snapshot) | `test_run_tracking.py::test_manifest_contains_required_fields` | `tests/test_utils/` |
| A31 | `append_run_history()` không đọc/ghi đè các dòng cũ, chỉ nối thêm | `test_run_tracking.py::test_run_history_append_only_preserves_old_rows` | `tests/test_utils/` |
| A32 | `list_runs()` trả về đúng thứ tự giảm dần theo thời gian, bỏ qua thư mục không đúng format run_id | `test_run_tracking.py::test_list_runs_sorted_and_filters_invalid_dirs` | `tests/test_utils/` |
| A33 | Ghi `manifest.json`/`latest_run.txt` là atomic — không để lại file `.tmp` nếu ghi bị ngắt giữa chừng | `test_run_tracking.py::test_atomic_write_leaves_no_partial_file` | `tests/test_utils/` |
| A34 | `load_predictions(run_id=None)` trả đúng dữ liệu của run mà `latest_run.txt` đang trỏ tới | `test_data_loader.py::test_data_loader_resolves_latest_run_by_default` | `tests/test_app/` |
| A35 | `load_predictions(run_id="<run cũ cụ thể>")` đọc đúng run đó, không lẫn với latest | `test_data_loader.py::test_data_loader_can_load_specific_historical_run` | `tests/test_app/` |
| A36 | Forward-fill macro chỉ áp dụng cho dòng thực sự NaN sau join (không ghi đè giá trị thật đã có) — MỞ RỘNG 2026-08-31: áp dụng cả Temperature/Fuel_Price, không chỉ CPI/Unemployment | `test_macro_forward_fill.py::test_forward_fill_does_not_overwrite_existing_values` | `tests/test_features/` |
| A37 | Forward-fill dùng đúng giá trị công bố gần nhất theo TỪNG Store (không lẫn giữa các Store, không dùng giá trị tương lai) | `test_macro_forward_fill.py::test_forward_fill_uses_last_known_value_per_store` | `tests/test_features/` |
| A38 | Flag `{col}_is_forward_filled` chỉ True đúng ở các dòng đã được điền, False ở mọi dòng khác (kể cả dòng NaN không điền được vì không có giá trị lịch sử nào) — MỞ RỘNG 2026-08-31: `test_forward_fill_applies_to_temperature_and_fuel_price` xác nhận Temperature/Fuel_Price cũng có flag | `test_macro_forward_fill.py::test_forward_fill_flag_matches_actually_filled_rows` | `tests/test_features/` |
| A39 | `aggregate_to_store_date` SUM đúng Weekly_Sales theo (Store,Date,IsHoliday). **DEPRECATED phần raise (2026-08-31):** IsHoliday không nhất quán trong cùng (Store,Date) từng phải raise `DataContractError` — GHI ĐÈ: nay tách thành nhiều dòng riêng theo IsHoliday, không raise — xem A47 | `test_aggregate_to_store_date.py::test_aggregate_sums_weekly_sales_across_dept`, `test_isholiday_inconsistent_splits_into_separate_rows` | `tests/test_ingestion/` |
| A40 | Sau aggregate, cột Dept không còn tồn tại ở bất kỳ đâu trong feature_matrix/predictions_long | `test_aggregate_to_store_date.py::test_dept_column_removed` | `tests/test_ingestion/` |
| A41 | Aggregate không cross-join giả định lưới đầy đủ Store × Date — số dòng sau aggregate = số (Store,Date) quan sát thật | `test_aggregate_to_store_date.py::test_aggregate_row_count_matches_observed_pairs` | `tests/test_ingestion/` |
| A42 | Lag/Rolling group theo `["Store"]` (không còn Dept) vẫn giữ đúng bất biến no-future-leak | `test_lag_rolling_no_future_leak.py` | `tests/test_features/` |
| A43 | `has_history` ở granularity Store: Store hoàn toàn mới ở test (không có trong train) được flag `False` đúng — bất biến cold-start logic vẫn hoạt động dù dữ liệu thật hiện không có ca này | `test_cold_start_handling.py::test_cold_start_store_flagged` | `tests/test_features/` |
| A44 | `NaiveSameWeekLastYear` group theo `["Store","week_of_year"]` (không Dept) dự đoán đúng, fallback global mean khi Store+week chưa từng thấy | `test_model_interface_consistency.py` | `tests/test_models/` |
| A45 | `load_enabled_blocks_from_config` đọc đúng danh sách block `enabled=true` từ `configs/features.yaml`, block `enabled=false` không xuất hiện trong kết quả | `test_feature_config_loading.py` | `tests/test_features/` |
| A46 | Gộp `train_window` + `valid_window`(không target) trước khi tính Lag/Rolling không làm thay đổi giá trị feature của `train_window` (bất biến chống leakage-ngược), và dòng đầu `valid_window` nhận đúng lag/rolling từ dữ liệu thật cuối `train_window` (không NaN oan) | `test_buffer_window_no_leakage.py` | `tests/test_features/` |
| A47 | `aggregate_to_store_date`/`join_features` group/join theo cả IsHoliday — IsHoliday lệch giữa các Dept cùng (Store,Date) tách thành nhiều dòng riêng (aggregate) hoặc tạo NaN ở cột features (join), không raise lỗi | `test_isholiday_inconsistent_splits_into_separate_rows`, `test_isholiday_mismatch_yields_nan_features_not_error` | `tests/test_ingestion/` |
| A48 | `add_markdown_features` fillna(0) cho MarkDown1-5, không còn tạo flag `has_markdown_{i}` | `test_markdown_flag.py::test_markdown_nan_becomes_zero` | `tests/test_features/` |
| A49 | `apply_macro_lag52_to_valid` ghi đúng giá trị macro từ 52 tuần trước theo từng Store, không lẫn Store khác, không có lịch sử 52 tuần trước thì trả NaN | `test_macro_lag52_valid.py` | `tests/test_features/` |
| A50 | `split_by_date_ratio` chia mốc theo tỷ lệ số NGÀY duy nhất (không phải số dòng), train luôn trước valid, không chồng lấn ngày | `test_ratio_split.py` | `tests/test_splitting/` |
| A51 | `load_lag_rolling_params_from_config` đọc đúng `lags`/`rolling_windows` từ `configs/features.yaml`; fallback dùng DEFAULT_* khi key không tồn tại trong YAML | `test_feature_config_loading.py::test_load_lag_rolling_params_reads_from_yaml`, `test_load_lag_rolling_params_falls_back_to_defaults_when_missing` | `tests/test_features/` |
| A52 | Đổi từ hard-code DEFAULT_LAGS/DEFAULT_ROLLING_WINDOWS sang đọc YAML không làm thay đổi kết quả `feature_matrix` khi giá trị YAML = giá trị default cũ (regression test) | `test_feature_config_loading.py::test_lag_rolling_params_match_previous_hardcoded_defaults` | `tests/test_features/` |

> Chi tiết đầy đủ skeleton của A16–A20 (Conformal Prediction): `08_uncertainty_conformal.md` mục 6. Chi tiết đầy đủ skeleton của A21–A23 (Dashboard): `07_dashboard_spec.md` mục 6. Chi tiết đầy đủ A25–A35 (Run tracking/versioning): `tests/test_utils/test_run_tracking.py`, `docs/00_decisions.md`. Chi tiết A36–A38 (Macro forward-fill), A39–A45 (Đổi đơn vị dự báo + features.yaml runtime) và A46 (Buffer nối train/valid): `docs/00_decisions.md` [2026-08-19]. Chi tiết A47–A52 (Đồng bộ xử lý dữ liệu theo notebook — aggregate/join theo IsHoliday, MarkDown fillna(0), macro ffill 4 cột, macro lag52 cho valid, split theo tỷ lệ 2/3, lags/rolling_windows từ config): `docs/00_decisions.md` [2026-08-31].

---

## 2. Skeleton test case chi tiết (pytest)

### 2.1. `tests/conftest.py` — fixture dữ liệu giả lập

```python
import pandas as pd
import pytest


@pytest.fixture
def sample_train():
    """Giả lập train.csv thu nhỏ: 2 Store, 2 Dept, 6 tuần, có 1 dòng sales âm."""
    return pd.DataFrame({
        "Store": [1, 1, 1, 1, 2, 2],
        "Dept": [1, 1, 1, 1, 1, 1],
        "Date": pd.to_datetime([
            "2010-02-05", "2010-02-12", "2010-02-19", "2010-02-26",
            "2010-02-05", "2010-02-12",
        ]),
        "Weekly_Sales": [100.0, 120.0, -5.0, 130.0, 200.0, 210.0],
        "IsHoliday": [False, True, False, False, False, True],
    })


@pytest.fixture
def sample_test():
    """(Store=3, Dept=1) chưa từng xuất hiện trong sample_train -> mô phỏng cold-start."""
    return pd.DataFrame({
        "Store": [1, 3],
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2010-03-05", "2010-03-05"]),
        "IsHoliday": [False, False],
    })


@pytest.fixture
def sample_features():
    """features.csv giả lập: MarkDown NaN có chủ đích ở 1 dòng."""
    return pd.DataFrame({
        "Store": [1, 1, 2],
        "Date": pd.to_datetime(["2010-02-05", "2010-02-12", "2010-02-05"]),
        "Temperature": [42.3, 38.5, 40.0],
        "Fuel_Price": [2.5, 2.6, 2.5],
        "MarkDown1": [None, 500.0, None],
        "CPI": [211.1, 211.2, 210.0],
        "Unemployment": [8.1, 8.1, 7.9],
        "IsHoliday": [False, True, False],
    })
```

### 2.2. Data Contract & Join (A1–A4, A14)

```python
# tests/test_ingestion/test_schema_validation.py
import pandas as pd
from sales_forecast.ingestion.validators import validate_train_schema, DataContractError


def test_train_schema_valid(sample_train):
    """Giả định A1: schema đúng thì validate không raise."""
    validate_train_schema(sample_train)  # không raise


def test_train_schema_rejects_missing_column(sample_train):
    """Giả định A1: thiếu cột bắt buộc phải raise, không âm thầm bỏ qua."""
    broken = sample_train.drop(columns=["IsHoliday"])
    with pytest.raises(DataContractError):
        validate_train_schema(broken)


def test_negative_sales_preserved(sample_train):
    """Giả định A14: Weekly_Sales âm KHÔNG bị clip về 0 khi ingest."""
    validate_train_schema(sample_train)
    assert (sample_train["Weekly_Sales"] < 0).any(), \
        "Fixture phải giữ ít nhất 1 giá trị âm để test có ý nghĩa"
    # Sau khi qua bước load thật (loaders.load_train), giá trị âm phải còn nguyên
```

```python
# tests/test_ingestion/test_join_integrity.py
from sales_forecast.ingestion.loaders import join_features


def test_no_cartesian_assumption(sample_train):
    """Giả định A2: pipeline KHÔNG được giả định lưới đầy đủ Store x Dept.
    Số tổ hợp (Store, Dept) thực tế phải khớp đúng dữ liệu quan sát, không suy diễn thêm."""
    observed_pairs = sample_train[["Store", "Dept"]].drop_duplicates()
    assert len(observed_pairs) == 2  # (1,1) và (2,1) — không phải 4 = 2 Store x 2 Dept giả định


def test_features_join_row_count(sample_train, sample_features):
    """Giả định A3: join theo (Store, Date) không được nhân bản dòng của train."""
    joined = join_features(sample_train, sample_features)
    assert len(joined) == len(sample_train), \
        "Join features.csv theo (Store,Date) không được làm thay đổi số dòng của train"


def test_isholiday_consistency(sample_train, sample_features):
    """Giả định A4: IsHoliday ở train và features phải khớp nhau sau khi join,
    lệch nhau phải được phát hiện thay vì âm thầm lấy 1 nguồn."""
    joined = join_features(sample_train, sample_features)
    mismatches = joined[joined["IsHoliday_train"] != joined["IsHoliday_features"]]
    assert len(mismatches) == 0, f"Phát hiện {len(mismatches)} dòng lệch IsHoliday giữa 2 nguồn"
```

### 2.3. Temporal Split & No-Leakage (A5, A6)

```python
# tests/test_splitting/test_temporal_split_no_leakage.py
import pandas as pd
from sales_forecast.splitting.temporal_split import temporal_split


def test_train_window_before_valid_window(sample_train):
    """Giả định A5: mọi Date trong train_window phải < mọi Date trong valid_window."""
    split_date = pd.Timestamp("2010-02-19")
    train_w, valid_w, _ = temporal_split(sample_train, split_date=split_date, horizon_weeks=1)
    assert train_w["Date"].max() < valid_w["Date"].min()


def test_valid_window_does_not_leak_into_train(sample_train):
    """Giả định A5: không có bản ghi nào xuất hiện ở cả train_window và valid_window."""
    split_date = pd.Timestamp("2010-02-19")
    train_w, valid_w, _ = temporal_split(sample_train, split_date=split_date, horizon_weeks=1)
    overlap = pd.merge(train_w, valid_w, on=["Store", "Dept", "Date"], how="inner")
    assert len(overlap) == 0
```

```python
# tests/test_features/test_lag_rolling_no_future_leak.py
import pandas as pd
from sales_forecast.features.lag_rolling import add_lag_features


def test_lag_feature_uses_only_past_data(sample_train):
    """Giả định A6: với mỗi dòng có Date = t, cột lag_1w phải bằng
    Weekly_Sales của đúng 1 tuần trước đó (Date = t - 7 ngày), KHÔNG được
    bằng giá trị của chính dòng t hay dòng tương lai."""
    df = add_lag_features(sample_train, group_cols=["Store", "Dept"], lags=[1])
    row = df[(df.Store == 1) & (df.Dept == 1) & (df.Date == "2010-02-12")].iloc[0]
    expected = sample_train[
        (sample_train.Store == 1) & (sample_train.Dept == 1) & (sample_train.Date == "2010-02-05")
    ]["Weekly_Sales"].iloc[0]
    assert row["lag_1w"] == expected


def test_first_observation_has_nan_lag_not_zero(sample_train):
    """Giả định A6 (liên quan A8): dòng đầu tiên của 1 chuỗi phải có lag = NaN,
    KHÔNG được điền 0 ở bước tạo feature (fillna là quyết định tường minh ở bước sau)."""
    df = add_lag_features(sample_train, group_cols=["Store", "Dept"], lags=[1])
    first_row = df[(df.Store == 1) & (df.Dept == 1)].sort_values("Date").iloc[0]
    assert pd.isna(first_row["lag_1w"])
```

### 2.4. MarkDown flag & Cold-start (A7, A8)

```python
# tests/test_features/test_markdown_flag.py
from sales_forecast.features.markdown_promo import add_markdown_features


def test_markdown_nan_becomes_explicit_flag(sample_features):
    """Giả định A7: NaN ở MarkDown1 phải tạo has_markdown=False,
    KHÔNG được lặng lẽ fillna(0) khiến 'không có dữ liệu' trông giống 'khuyến mãi = 0đ'."""
    df = add_markdown_features(sample_features)
    nan_rows = df[df["MarkDown1"].isna()]
    assert (nan_rows["has_markdown_1"] == False).all()
```

```python
# tests/test_features/test_cold_start_handling.py
from sales_forecast.features.pipeline import build_feature_matrix


def test_cold_start_pair_flagged(sample_train, sample_test, sample_features):
    """Giả định A8: (Store=3, Dept=1) không có trong train phải được đánh dấu
    has_history=False, KHÔNG được crash pipeline và KHÔNG được âm thầm dùng
    lag từ Store/Dept khác."""
    fm = build_feature_matrix(sample_train, sample_test, sample_features)
    cold_row = fm[(fm.Store == 3) & (fm.Dept == 1)]
    assert len(cold_row) == 1
    assert cold_row["has_history"].iloc[0] == False
    assert pd.isna(cold_row["lag_1w"].iloc[0])
```

### 2.5. Feature block độc lập (A9)

```python
# tests/test_features/test_feature_block_independence.py
from sales_forecast.features.pipeline import build_feature_matrix


def test_disabling_markdown_block_does_not_break_pipeline(sample_train, sample_test, sample_features):
    """Giả định A9: tắt block MarkDown qua config không được làm hỏng các block khác
    (Lag/Calendar/Encoding vẫn phải chạy đúng)."""
    fm_full = build_feature_matrix(sample_train, sample_test, sample_features,
                                    enabled_blocks=["lag_rolling", "calendar", "markdown", "encoding"])
    fm_no_markdown = build_feature_matrix(sample_train, sample_test, sample_features,
                                           enabled_blocks=["lag_rolling", "calendar", "encoding"])
    assert "MarkDown1" not in fm_no_markdown.columns
    assert "lag_1w" in fm_no_markdown.columns
    assert set(fm_no_markdown["Store"]) == set(fm_full["Store"])
```

### 2.6. Evaluation Layer & Metric (A10, A11)

```python
# tests/test_evaluation/test_cv_splitter_order.py
from sales_forecast.evaluation.cv_splitter import walk_forward_splits


def test_walk_forward_never_puts_future_in_train(sample_train):
    """Giả định A10: với mọi fold, max(Date) của train phải < min(Date) của valid."""
    for train_idx, valid_idx in walk_forward_splits(sample_train, n_splits=2, date_col="Date"):
        train_dates = sample_train.iloc[train_idx]["Date"]
        valid_dates = sample_train.iloc[valid_idx]["Date"]
        assert train_dates.max() < valid_dates.min()
```

```python
# tests/test_evaluation/test_metrics_correctness.py
import numpy as np
from sales_forecast.evaluation.metrics import weighted_mae


def test_weighted_mae_holiday_weight_is_5x():
    """Giả định A11: tuần lễ (IsHoliday=True) phải có trọng số gấp 5 lần
    tuần thường trong công thức WMAE, đúng luật đánh giá gốc Kaggle."""
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 90.0])   # sai số 10 ở cả 2 dòng
    is_holiday = np.array([False, True])
    wmae = weighted_mae(y_true, y_pred, is_holiday)
    # dòng holiday đóng góp gấp 5 lần dòng thường -> wmae != mae thường (=10)
    plain_mae = np.mean(np.abs(y_true - y_pred))
    assert wmae != plain_mae
    expected = (10 * 1 + 10 * 5) / (1 + 5)
    assert np.isclose(wmae, expected)
```

### 2.7. Model interface & Optuna leakage (A12, A13)

```python
# tests/test_models/test_model_interface_consistency.py
import pytest
from sales_forecast.models.registry import get_model


@pytest.mark.parametrize("model_name", ["naive_baseline", "decision_tree", "random_forest", "lightgbm", "xgboost"])
def test_model_has_unified_interface(model_name):
    """Giả định A12: mọi model, kể cả baseline, phải expose đúng .fit(X, y) / .predict(X)
    để Evaluation Layer dùng chung 1 đoạn code, không if/else riêng theo loại model."""
    model = get_model(model_name)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
```

```python
# tests/test_models/test_optuna_no_test_leakage.py
from sales_forecast.models.tuning import run_optuna_study


def test_optuna_objective_never_receives_test_window(sample_train, monkeypatch):
    """Giả định A13: hàm objective của Optuna chỉ được truyền train_window/valid_window,
    test_window không được xuất hiện trong closure của objective function."""
    # Thiết kế: assert bằng cách kiểm tra study.trials không tham chiếu test_window
    # (spy/mock objective để xác nhận tham số truyền vào không chứa dữ liệu test)
    ...  # triển khai cụ thể phụ thuộc chữ ký hàm run_optuna_study thật
```

### 2.8. SHAP stability (A15)

```python
# tests/test_explainability/test_shap_stability.py
from sales_forecast.explainability.stability import compare_shap_rank_across_folds


def test_shap_rank_correlation_reported_not_single_shot():
    """Giả định A15: pipeline phải trả về so sánh rank importance qua >= 2 fold
    (vd. Spearman correlation), KHÔNG được chỉ tính SHAP 1 lần trên toàn bộ data
    rồi coi là kết luận cuối cùng."""
    fold_importances = [
        {"lag_1w": 0.30, "MarkDown1": 0.10, "Store": 0.05},
        {"lag_1w": 0.28, "MarkDown1": 0.12, "Store": 0.04},
    ]
    result = compare_shap_rank_across_folds(fold_importances)
    assert "rank_correlation" in result
    assert -1.0 <= result["rank_correlation"] <= 1.0
```

---

## 3. Quy ước chạy test

```bash
# Cài đặt (editable) trước khi test
pip install -e ".[dev]"

# Chạy toàn bộ test
pytest tests/ -v

# Chạy riêng nhóm chống leakage (nhóm quan trọng nhất, nên chạy trước mỗi commit)
pytest tests/test_splitting tests/test_features -k "leak" -v

# Chạy riêng nhóm conformal prediction (trước khi đụng vào 06/07/08)
pytest tests/test_evaluation -k "conformal" -v

# Chạy riêng nhóm dashboard (không cần data thật, chỉ test hợp đồng dữ liệu + logic trình bày)
pytest tests/test_app -v

# Bắt buộc trong CI / pre-commit trước khi merge vào nhánh chính
pytest tests/ --cov=src/sales_forecast --cov-report=term-missing
```

**Ngưỡng tối thiểu đề xuất:** coverage ≥ 80% cho `src/sales_forecast/ingestion`, `splitting`, `features`, `evaluation/conformal.py` (nhóm rủi ro leakage cao nhất); các module `models`/`explainability` ≥ 60%; `app/` ≥ 50% (chủ yếu logic load/format dữ liệu, không cần cover UI Streamlit chi tiết).

## 4. Khi thêm giả định mới

Mỗi khi team đưa ra một quyết định kiến trúc mới trong `docs/00_decisions.md`, **bắt buộc** bổ sung ngay 1 dòng vào bảng mục 1 của tài liệu này kèm test tương ứng trước khi merge — tránh nợ kỹ thuật kiểu "để test sau".
