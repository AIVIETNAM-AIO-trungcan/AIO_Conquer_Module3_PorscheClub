# Báo cáo tiến độ Tuần 1.2 — Xử lý biến ngoại sinh & Đổi đơn vị dự báo

**AIO Conquer 2026 — Module 03 · Project 3.2**

> Ngày báo cáo: 2026-08-19. Bối cảnh: sau buổi thảo luận nhóm tiếp theo buổi baseline tuần 1, team phát hiện 2 vấn đề kiến trúc cần xử lý trước khi bước sang Giai đoạn 5 (Model Training đầy đủ) — (1) cách hiểu về biến ngoại sinh (Temperature, CPI, Fuel_Price, MarkDown...) trong `test_window`, (2) mong muốn giới hạn phạm vi dự đoán bằng cách gom nhóm Dept lại theo Store.

---

## 1. Tóm tắt tình trạng

**Đã hoàn tất cả 2 thay đổi kiến trúc.** Thay đổi 1 (fallback CPI/Unemployment) độc lập, rủi ro thấp, xong trước. Thay đổi 2 (đổi đơn vị dự báo từ `(Store, Dept, Date)` sang `(Store, Date)`) là thay đổi lớn, xuyên suốt nhiều module — đã **ghi đè có chủ đích** quyết định đơn vị dự báo chốt ngày 2026-08-18, giữ nguyên lịch sử quyết định cũ trong `docs/00_decisions.md`. Toàn bộ 46/46 test tự động pass (tăng từ 20/20 ở tuần 1), pipeline chạy thành công end-to-end trên dữ liệu thật. Hệ quả quan trọng cần lưu ý: project không còn tạo được `submission.csv` đúng format nộp Kaggle gốc (xem mục 3).

---

## 2. Đã hoàn thành

### Thay đổi 1 — Fallback CPI/Unemployment ở đuôi test_window

**Phát hiện qua khảo sát trực tiếp `data/raw/features.csv`:** file này thực ra phủ hết `test_window` (đến 2013-07-26), không dừng ở cuối train như tài liệu cũ ngầm giả định (`docs/03_data_io_diagram.md` trước đó ghi "cần xác nhận độ trễ công bố trong EDA"). Temperature/Fuel_Price/MarkDown1-5 gần như đầy đủ trong toàn bộ test_window — đây LÀ biến ngoại sinh quan sát được tại thời điểm t, không cần tự dự báo. Riêng CPI/Unemployment thiếu **đúng 585 dòng = 13 tuần cuối test_window (2013-05-03 → 2013-07-26) ở cả 45 Store**, pattern giống hệt nhau giữa 2 cột — do độ trễ công bố macro index thật, không phải lỗi ngẫu nhiên.

- `src/sales_forecast/features/macro.py`: thêm `_add_macro_forward_fill()` — forward-fill CPI/Unemployment theo từng Store (dùng giá trị công bố gần nhất trước đó, không nhìn tương lai), kèm flag tường minh `cpi_is_forward_filled`/`unemployment_is_forward_filled` (bool), nhất quán triết lý `has_markdown` đã chốt — không fillna mù quáng. Không hard-code cửa sổ "13 tuần" trong code — `ffill` tự nhiên chỉ điền đúng chỗ có NaN.
- Cập nhật `docs/03_data_io_diagram.md`, `docs/01_ideation.md` mục 2.3 — chuyển từ "cần xác nhận" sang khẳng định dứt khoát kèm ngoại lệ.
- Ghi quyết định vào `docs/00_decisions.md` [2026-08-19] "Xử lý CPI/Unemployment missing ở đuôi test_window".
- 4/4 test mới pass (`tests/test_features/test_macro_forward_fill.py`) — A36-A38 trong `docs/05_test_plan.md`.

### Thay đổi 2 — Đổi đơn vị dự báo (Store, Dept, Date) → (Store, Date)

**Lý do (xác nhận với team):** (1) giảm độ phức tạp mô hình hóa — 81 Dept/Store tạo ~2.000+ chuỗi ngắn/thưa, chi phí tính toán/tuning lớn hơn đáng kể so với 45 chuỗi Store; bằng chứng thực nghiệm ở báo cáo tuần 1 mục 3.2 (Decision Tree thua Naive do ordinal encoding không đủ cho quá nhiều tổ hợp Store-Dept) càng củng cố hướng này. (2) mục tiêu dashboard/báo cáo tập trung cấp Store dễ diễn giải hơn cho người xem cuối.

**Khảo sát dữ liệu ủng hộ quyết định:** mỗi (Store, Date) có đầy đủ 45 Store × mọi tuần (6435 combos = 45 × 143 tuần, khớp chính xác); `IsHoliday` nhất quán 100% trong mọi (Store, Date); **không có Store nào cold-start ở test** (khác hẳn cold-start 11 cặp Store-Dept ở granularity cũ); sau khi aggregate SUM theo Dept thì không còn dòng Weekly_Sales âm nào (min = 209,986, do trung bình ~65 Dept/Store-week cộng lại lấn át các dòng âm lẻ tẻ).

**Thay đổi code (theo đúng thứ tự thực hiện):**

| Module | Thay đổi |
|---|---|
| `src/sales_forecast/ingestion/schema.py` | Thêm `train_aggregated_schema`/`test_aggregated_schema` (không Dept); schema raw gốc giữ nguyên Dept vì vẫn validate đúng CSV gốc trước aggregate |
| `src/sales_forecast/ingestion/loaders.py` | Hàm mới `aggregate_to_store_date()` — SUM Weekly_Sales theo (Store, Date), **raise `DataContractError`** nếu IsHoliday không nhất quán giữa các Dept cùng Store-tuần (tránh `.first()` âm thầm chọn sai) |
| `src/sales_forecast/ingestion/validators.py` | `validate_train_aggregated_schema`/`validate_test_aggregated_schema` |
| `src/sales_forecast/features/store_encoding.py` | Đổi tên từ `store_dept_encoding.py`, hàm `encode_store` thay `encode_store_dept` |
| `src/sales_forecast/features/pipeline.py` | `_add_has_history`, `build_feature_matrix` group_cols đổi sang `["Store"]`; **thêm `load_enabled_blocks_from_config()`** — đọc `configs/features.yaml` runtime thay vì truyền cứng Python list (gap kỹ thuật có sẵn từ trước, được gộp fix cùng lần này) |
| `src/sales_forecast/models/baseline.py` | `NaiveSameWeekLastYear` groupby/index đổi sang `["Store", "week_of_year"]` |
| `pipelines/run_train_baseline.py` | Thêm bước "1b. Aggregate" ngay sau validate schema raw, trước Temporal Split; đọc `enabled_blocks` từ config |
| `configs/data.yaml`, `configs/features.yaml` | Bỏ `dept_column`; đổi key block `store_dept_encoding` → `store_encoding` |

**Cập nhật tài liệu:** `docs/00_decisions.md` (entry ghi đè, không xóa quyết định cũ 2026-08-18), `docs/01_ideation.md`, `docs/02_pipeline_architecture.md` (thêm node "1b. Aggregate"), `docs/03_data_io_diagram.md` (ERD, Feature Matrix, `predictions_long` bỏ Dept và `is_cold_start`, `product_age_weeks` xóa khỏi đặc tả), `docs/07_dashboard_spec.md` (bỏ filter "Chọn Dept"), `docs/05_test_plan.md` (A8 đánh dấu DEPRECATED, thêm A39-A45).

**Test:** 5/5 test mới cho `aggregate_to_store_date` (`tests/test_ingestion/test_aggregate_to_store_date.py`), test cho `load_enabled_blocks_from_config` (`tests/test_features/test_feature_config_loading.py`), toàn bộ test hiện có bị ảnh hưởng bởi đổi interface đã được sửa (`test_cold_start_handling.py`, `test_lag_rolling_no_future_leak.py`, `test_temporal_split_no_leakage.py`, `test_model_interface_consistency.py`), thêm fixture `sample_train_aggregated`/`sample_test_aggregated` vào `conftest.py`.

### Tổng kết kỹ thuật

- **46/46 test pass** (`pytest tests/ -v`), tăng từ 20/20 tuần 1
- Coverage tổng `src/sales_forecast`: **88%** (`ingestion` 72-86%, `features` 100%, `models` 95%, `evaluation` 100%, `utils` 74-99%)
- `python scripts/check_env.py` → PASS
- Chạy pipeline thật end-to-end (`pipelines/run_train_baseline.py`) trên `data/raw/` thành công, `data/raw/` không bị ghi đè (đã xác nhận checksum không đổi)
- Không có leakage thời gian nào bị phát hiện; mọi tham số đọc từ `configs/`, không hard-code (kể cả cửa sổ forward-fill 13 tuần)

**Kết quả baseline trên granularity mới** (train_window 4,500 dòng / valid_window 1,935 dòng, horizon 52 tuần — số dòng nhỏ hơn hẳn tuần 1 vì đã aggregate từ ~65 Dept/Store-week xuống 1 dòng/Store-week):

| Model | WMAE | WMAPE |
|---|---|---|
| Naive same-week-last-year | 66,469.94 | 0.0689 (6.89%) |
| Decision Tree đơn giản | 455,746.45 | 0.6596 |

> **Không so sánh trực tiếp được với số liệu WMAE tuần 1** (2,217.87 / 14,199.97) — đơn vị đo Weekly_Sales đã đổi hoàn toàn do SUM theo Dept. Điểm đáng chú ý: WMAPE giờ ở mức hợp lý (6.89%) thay vì bùng nổ hàng chục triệu % như tuần 1 — vì aggregate loại bỏ hết các dòng sales gần 0 (0 dòng `|Weekly_Sales| < 1.0` trong valid_window, so với 345 dòng ở tuần 1). Đây là tác dụng phụ tích cực của việc đổi granularity, không phải mục tiêu chính.

---

## 3. Hệ quả cần lưu ý

### 3.1. Mất khả năng nộp bài đúng format Kaggle gốc (đã xác nhận với team, chấp nhận đánh đổi)

`submission.csv` giờ chỉ còn (Store, Date, Weekly_Sales dự báo) — không còn Dept trong id `Store_Dept_Date` theo đúng luật thi Kaggle gốc. Project từ thời điểm này **không còn mục tiêu đối chiếu Kaggle leaderboard công khai**. Đối tượng bàn giao là dashboard Streamlit + báo cáo kỹ thuật nội bộ, đúng yêu cầu cốt lõi của Module 03 (thể hiện quy trình ML hoàn chỉnh và khả năng giải thích kết quả).

### 3.2. Cold-start Dept-level (A8) không còn áp dụng

11 cặp Store-Dept cold-start từng ghi nhận ở tuần 1 không còn ý nghĩa ở granularity (Store, Date) — đã xác nhận không có Store nào cold-start trên dữ liệu thật (cả 45 Store đều có trong train). `has_history` được giữ lại trong code như bất biến phòng thủ (defensive) — nếu tương lai xuất hiện Store hoàn toàn mới, flag vẫn phải hoạt động đúng, đã có test riêng xác nhận (A43).

### 3.3. Gap kỹ thuật đã fix kèm theo (ngoài phạm vi yêu cầu gốc)

Phát hiện trong lúc rà soát: `configs/features.yaml` trước đây không được code nào đọc runtime — `enabled_blocks` truyền cứng qua Python list ở nơi gọi. Đã bổ sung `load_enabled_blocks_from_config()` để `run_train_baseline.py` đọc đúng danh sách block bật/tắt từ file YAML, đúng tinh thần CLAUDE.md mục 4 "bật/tắt qua config, không hard-code trong src/".

---

## 4. Đối chiếu mục tiêu Tuần 1.2

| Target | Đã đạt | Ghi chú |
|---|---|---|
| Fix cách hiểu/xử lý biến ngoại sinh trong test_window | ✅ Đạt | Xác nhận features.csv phủ hết test_window; forward-fill cho CPI/Unemployment 13 tuần cuối, kèm flag tường minh |
| Đổi đơn vị dự báo sang (Store, Date) | ✅ Đạt | Toàn bộ pipeline (schema, ingestion, feature, model, orchestration, config) đã đồng bộ |
| Không phá vỡ pipeline hiện có | ✅ Đạt | 46/46 test pass, pipeline end-to-end chạy được, coverage duy trì 88% |
| Ghi lại quyết định kiến trúc đầy đủ | ✅ Đạt | 2 entry mới trong `docs/00_decisions.md`, giữ nguyên lịch sử quyết định cũ, không xóa |
| Cập nhật tài liệu liên quan | ✅ Đạt | 7 file docs đã rà soát và cập nhật (ideation, pipeline architecture, data I/O, dashboard spec, test plan) |

**Kết luận: ĐẠT mục tiêu Tuần 1.2.** Không phát sinh hạn chế kỹ thuật mới cần theo dõi — điểm cần lưu ý duy nhất là hệ quả đã biết trước và được team chấp nhận (mục 3.1).

---

## 5. Kế hoạch tiếp theo (đề xuất)

Quay lại đúng lộ trình Giai đoạn 5 đã đề ra ở báo cáo tuần 1, trên nền granularity (Store, Date) mới:

1. **Giai đoạn 5** — Model Training đầy đủ: Decision Tree → Random Forest → AdaBoost/GBM → LightGBM (chính) → XGBoost, tất cả qua cùng Evaluation Layer, trên đơn vị dự báo (Store, Date)
2. **Giai đoạn 6** — Optuna Tuning, tái dùng walk-forward split của Evaluation Layer
3. **Giai đoạn 6b** — Conformal Calibration (Split Conformal 95%, tỷ lệ calib_window đã chốt 50/50) — không bị ảnh hưởng bởi 2 thay đổi lần này (cả hai đều ở giai đoạn 1-3, trước giai đoạn 6b)
4. **Giai đoạn 7-9** — Evaluation & Error Analysis (theo Store thay vì Dept), TreeSHAP, Reporting & Dashboard Streamlit (đã cập nhật spec bỏ filter Dept)
