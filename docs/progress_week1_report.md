# Báo cáo tiến độ Tuần 1 — Baseline

**AIO Conquer 2026 — Module 03 · Project 3.2**

> Ngày báo cáo: 2026-08-18. Mục tiêu team đặt ra: hết tuần 1 hoàn tất baseline.
>
> **Lưu ý (cập nhật 2026-08-19):** sau báo cáo này, đơn vị dự báo đã đổi từ `(Store, Dept, Date)` sang `(Store, Date)` — xem `docs/00_decisions.md` "Đổi đơn vị dự báo". Toàn bộ số liệu WMAE/WMAPE baseline trong báo cáo này (2,217.87 / 14,199.97 ở mục 2 Giai đoạn 4) đo trên granularity Store-Dept cũ, **không còn so sánh trực tiếp được** với kết quả chạy pipeline sau ngày 2026-08-19 (đơn vị đo Weekly_Sales đã đổi hoàn toàn do aggregate SUM theo Dept). Giữ nguyên nội dung báo cáo gốc bên dưới làm lịch sử.

---

## 1. Tóm tắt tình trạng

**Đã đạt mục tiêu tuần 1.** Pipeline chạy được từ đầu đến cuối trên dữ liệu thật (`data/raw/`), có baseline (Naive same-week-last-year và Decision Tree đơn giản) đo được kết quả bằng WMAE/WMAPE, có 20/20 test tự động pass. Còn 2 điểm cần lưu ý — không chặn tiến độ, nhưng cần theo dõi ở tuần 2 (xem mục 3-4).

---

## 2. Đã hoàn thành

### Giai đoạn 0 — Problem Framing

Toàn bộ quyết định nền tảng đã chốt vào `docs/00_decisions.md`:

| Hạng mục | Quyết định |
|---|---|
| Forecast horizon & chiến lược | Direct multi-step, 39 tuần, chia 3 nhóm horizon (h=1-4, h=5-12, h=13-39) |
| Weekly_Sales âm | Giữ nguyên, không clip |
| Cold-start (11 cặp Store-Dept) | Flag `has_history=False` + fallback theo Dept/Type |
| Metric chính | Báo cáo song song WMAE **và** WMAPE, cùng trọng số IsHoliday x5 |
| Tỷ lệ `calib_window`/`valid_window` | 50/50 (dùng cho Conformal Prediction ở giai đoạn sau) |
| WMAPE với sales gần 0 | Giữ nguyên công thức chuẩn, không loại trừ dòng (xem mục 3) |

### Giai đoạn 1 — Data Ingestion & Validation

- `src/sales_forecast/ingestion/`: Data Contract bằng `pandera` (`schema.py`), `DataContractError` raise rõ ràng khi sai schema (`validators.py`), `load_raw_data` + `join_features` (`loaders.py`)
- Verify trên dữ liệu thật: Data Contract **PASS** trên toàn bộ 421,570 dòng `train.csv`; join với `features.csv` theo (Store, Date) không nhân bản/mất dòng; 1,285 dòng Weekly_Sales âm được giữ nguyên đúng quyết định đã chốt
- 6/6 test pass (`tests/test_ingestion/`)

### Giai đoạn 2 — Temporal Split

- `src/sales_forecast/splitting/temporal_split.py`: cắt `train_window`/`valid_window` theo `as_of_date`, không dùng random K-Fold
- 2/2 test pass, xác nhận không leakage (train luôn kết thúc trước valid bắt đầu)

### Giai đoạn 3 — Feature Engineering (5 block độc lập)

`src/sales_forecast/features/`: `lag_rolling.py`, `calendar.py`, `markdown_promo.py`, `store_dept_encoding.py`, `macro.py`, ghép bởi `pipeline.py` (`build_feature_matrix`), sinh thêm cột `has_history` cho cold-start.

- Lag/rolling: dòng đầu chuỗi = NaN (không fillna 0), chỉ dùng dữ liệu ≤ t−1
- MarkDown: flag `has_markdown_{1..5}` tường minh, giữ nguyên NaN gốc
- Mỗi block bật/tắt độc lập, không phá vỡ block khác khi tắt 1 block
- 5/5 test pass (`tests/test_features/`)

### Metrics — Evaluation cơ bản

- `src/sales_forecast/evaluation/metrics.py`: `weighted_mae`, `weighted_mape` (cả hai cùng trọng số IsHoliday x5)
- 3/3 test pass (`tests/test_evaluation/`)

### Giai đoạn 4 — Baseline Models

`src/sales_forecast/models/baseline.py`: `NaiveSameWeekLastYear`, `SimpleDecisionTreeBaseline`, cùng interface `.fit(X, y)/.predict(X)`.

Chạy end-to-end trên dữ liệu thật (`pipelines/run_train_baseline.py`, train_window 294,132 dòng / valid_window 127,438 dòng, horizon 52 tuần):

| Model | WMAE | WMAPE |
|---|---|---|
| Naive same-week-last-year | **2,217.87** | 96,728,347.38 (xem mục 3) |
| Decision Tree đơn giản | 14,199.97 | 27,633,768.62 (xem mục 3) |

- 4/4 test pass (`tests/test_models/`)

### Tổng kết kỹ thuật

- **20/20 test pass** (`pytest tests/ -v`)
- `python scripts/check_env.py` → PASS
- Không có leakage thời gian nào bị phát hiện; mọi tham số đọc từ `configs/`, không hard-code

---

## 3. Hạn chế phát hiện được

### 3.1. WMAPE bùng nổ do Weekly_Sales gần 0 (đã ghi nhận, không chặn tiến độ)

Khi chạy trên dữ liệu thật, WMAPE cho ra giá trị bất thường (hàng chục triệu %) trong khi WMAE vẫn hợp lý. Nguyên nhân: 345 dòng trong `valid_window` có `|Weekly_Sales| < 1.0` (nhỏ nhất 0.01, ví dụ Store 1 – Dept 99). Với các dòng này, `APE = |y_true − y_pred| / |y_true|` cho giá trị cực lớn dù sai số tuyệt đối rất nhỏ, làm trung bình có trọng số trên toàn bộ ~127K dòng bị một số ít dòng chi phối.

Đây là **hạn chế toán học đã biết của MAPE**, không phải lỗi implementation — đã được xác nhận qua điều tra thủ công và ghi quyết định xử lý vào `docs/00_decisions.md`.

### 3.2. Decision Tree baseline kém hơn Naive baseline (đã điều tra, giới hạn thuật toán — xem tuần 2)

| Model | WMAE |
|---|---|
| Naive same-week-last-year | 2,217.87 |
| Decision Tree đơn giản | 14,199.97 (kém hơn ~6.4 lần) |

**Cập nhật sau điều tra kỹ hơn:** ban đầu nghi ngờ `SimpleDecisionTreeBaseline.fit()` dùng `X.select_dtypes(include="number")` âm thầm loại bỏ `Store`/`Dept` (dtype `category` do block `store_dept_encoding` tạo ra). Đã sửa `fit()`/`predict()` để convert cột `category` sang mã số (`.cat.codes`) trước khi lọc numeric, thay vì loại bỏ hoàn toàn — đây là sửa đúng cần thiết (loại bỏ âm thầm 1 cột định danh luôn là lỗi, bất kể ảnh hưởng kết quả).

**Tuy nhiên WMAE sau khi sửa không đổi (vẫn 14,199.97)**, vì nguyên nhân thật sâu hơn: kiểm tra `model.feature_importances_` cho thấy `Store`/`Dept` có importance ≈ 0.0000 ngay cả sau khi sửa, dù cột đã có mặt trong tập feature. Đã thử tăng `max_depth` (3 → 8) để cây có đủ độ sâu tách theo `Store`/`Dept` — kết quả WMAE **tệ hơn** (14,875.13), vì `.cat.codes` là ordinal encoding không có thứ tự nghiệp vụ thật (mã số 1, 2, 3... không phản ánh mức doanh số liên tục); cây sâu hơn chỉ overfit theo thứ tự giả tạo đó thay vì generalize tốt hơn trên `valid_window`.

**Kết luận: đây là giới hạn cấu trúc của `sklearn.DecisionTreeRegressor`**, vốn không hỗ trợ categorical native (khác LightGBM) — không phải bug có thể sửa bằng cách chỉnh tham số hoặc lọc cột. Với ~2,000+ tổ hợp Store-Dept, ordinal encoding không đủ để một cây đơn (dù nông hay sâu) tái tạo hành vi tra cứu chính xác như Naive. `max_depth` đã revert về 3 (giá trị gốc); giải pháp thật sự đúng hướng là chuyển sang LightGBM ở Giai đoạn 5 (đã có trong kế hoạch tuần 2, mục 6), nơi `Store`/`Dept` được xử lý categorical native.

---

## 4. Giải pháp khắc phục đề xuất

| Hạn chế | Giải pháp | Mức độ ưu tiên |
|---|---|---|
| WMAPE bùng nổ (3.1) | Không sửa code. Diễn giải WMAPE luôn kèm WMAE, không dùng riêng lẻ. Thêm lát cắt riêng "sales gần 0 vs. bình thường" trong Giai đoạn 7 (Evaluation & Error Analysis) | Thấp — đã có kế hoạch xử lý đúng chỗ trong pipeline |
| Decision Tree thua Naive (3.2) | Đã sửa `select_dtypes` (không còn âm thầm loại bỏ Store/Dept) — nhưng đã kiểm chứng đây là giới hạn thuật toán của sklearn tree với ordinal encoding, không phải bug đơn thuần. Không tiếp tục vá thêm baseline (target encoding sẽ thêm rủi ro leakage cho 1 model chỉ mang tính minh họa); chuyển hướng đúng đắn sang LightGBM (categorical native) ở Giai đoạn 5 | Thấp — không chặn Giai đoạn 5, đã có hướng xử lý đúng chỗ (đổi model, không đổi baseline) |

---

## 5. Đối chiếu mục tiêu tuần 1

| Target team đề ra | Đã đạt | Ghi chú |
|---|---|---|
| Hoàn tất baseline | ✅ Đạt | Naive + Decision Tree chạy được trên dữ liệu thật, đo được WMAE/WMAPE |
| Pipeline không leakage | ✅ Đạt | Temporal Split trước Feature Engineering, 20/20 test pass gồm các test chống leakage (A5, A6) |
| Có test coverage cho giả định quan trọng | ✅ Đạt | A1-A9, A11, A12 đã có test (theo `docs/05_test_plan.md`) |
| Baseline chất lượng cao | ⚠️ Một phần | Naive baseline hợp lý; Decision Tree cần sửa encoding ở tuần 2 (mục 3.2) — không phải yêu cầu bắt buộc của tuần 1 |

**Kết luận: ĐẠT mục tiêu tuần 1**, với 2 hạn chế đã ghi nhận rõ ràng và có hướng xử lý cụ thể cho tuần 2.

---

## 6. Kế hoạch tuần 2 (đề xuất)

Theo đúng thứ tự `docs/02_pipeline_architecture.md`:

1. **Giai đoạn 5** — Model Training đầy đủ: Decision Tree (sửa encoding) → Random Forest → AdaBoost/GBM → LightGBM (chính) → XGBoost → CatBoost, tất cả qua cùng Evaluation Layer
2. **Giai đoạn 6** — Optuna Tuning, tái dùng walk-forward split của Evaluation Layer
3. **Giai đoạn 6b** — Conformal Calibration (Split Conformal 95%, tỷ lệ calib_window đã chốt 50/50)
4. **Giai đoạn 7-9** — Evaluation & Error Analysis (bao gồm lát cắt "sales gần 0"), TreeSHAP, Reporting & Dashboard Streamlit
