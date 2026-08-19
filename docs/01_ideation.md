# Ideation — Sales & Demand Forecasting

**AIO Conquer 2026 — Module 03 · Project 3.2: Sales Forecasting and Demand Prediction Using LightGBM and SHAP**

> Phiên bản: rev2 — 2026-08-18
> rev1: dựa **hoàn toàn trên dữ liệu thật** đã có trong `data/raw/` (Walmart Recruiting - Store Sales Forecasting, Kaggle), không còn dùng M5 làm case-study minh họa.
> rev2 (bản này): bổ sung 3 yêu cầu kỹ thuật mới — (1) môi trường venv/pip tái lập được cho team & mentor, (2) dashboard Streamlit tổng hợp biểu đồ + metric, (3) khoảng dự đoán tin cậy 95% bằng Split Conformal Prediction. Chi tiết từng phần: `06_environment_setup.md`, `07_dashboard_spec.md`, `08_uncertainty_conformal.md`.

---

## 1. Bối cảnh & ràng buộc từ chương trình

| Hạng mục | Nội dung |
|---|---|
| Đề tài chính thức | Project 3.2 — Sales Forecasting and Demand Prediction Using LightGBM và SHAP |
| Mô hình cây bắt buộc | Decision Tree → Random Forest → AdaBoost → Gradient Boost → LightGBM (chính) / XGBoost (so sánh) |
| Feature engineering | Lag, Rolling window (Time-series) |
| Chia dữ liệu | TimeSeriesSplit — **bắt buộc**, không dùng random K-Fold |
| Tối ưu tham số | Optuna |
| Giải thích mô hình | TreeSHAP |
| Khoảng dự đoán tin cậy | 95%, Split Conformal Prediction (xem `08_uncertainty_conformal.md`) |
| Môi trường tái lập | venv + pip, `pyproject.toml` (xem `06_environment_setup.md`) |
| Kết quả bàn giao | Dashboard Streamlit tổng hợp (xem `07_dashboard_spec.md`) |
| Hạn nộp | 7/9 – 9/9/2026 (Peer Review 10–12/9, kiểm tra toàn bộ 13/9) |

---

## 2. Dữ liệu thật — kết quả khảo sát trực tiếp

Bộ dữ liệu trong `data/raw/` là **Walmart Recruiting - Store Sales Forecasting** (Kaggle, 2014), gồm 4 file. Đây là điểm khác biệt quan trọng nhất cần lưu ý: dữ liệu ở **tần suất tuần (weekly)**, không phải theo ngày.

### 2.1. `train.csv` — 421.570 dòng × 5 cột

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `Store` | int | 1–45 |
| `Dept` | int | 1–81 (không liên tục, không phải mọi Store có đủ 81 Dept) |
| `Date` | date | tuần, thứ Sáu là ngày mốc, 2010-02-05 → 2012-10-26 |
| `Weekly_Sales` | float | doanh số tuần của (Store, Dept). **Có 1.285 dòng âm** (trả hàng/điều chỉnh sổ sách) và 73 dòng bằng 0 |
| `IsHoliday` | bool | tuần có chứa ngày lễ trọng điểm (Super Bowl, Labor Day, Thanksgiving, Christmas) |

- Số chuỗi (Store × Dept) xuất hiện trong train: **3.331 chuỗi**, không phải lưới đầy đủ 45 × 81 = 3.645 — tức là **không phải mọi Store đều bán mọi Dept**, đây là tín hiệu missing-by-design chứ không phải lỗi dữ liệu.
- Không có giá trị null ở train.
- **Lưu ý (cập nhật 2026-08-19):** pipeline hiện tại aggregate SUM Weekly_Sales theo (Store, Date) ngay sau ingestion — số liệu 3.331 chuỗi Store × Dept ở trên mô tả đúng RAW data, không còn phản ánh đơn vị dự báo hiện tại. Xem `docs/00_decisions.md` [2026-08-19] "Đổi đơn vị dự báo: (Store, Dept, Date) -> (Store, Date)".

### 2.2. `test.csv` — 115.064 dòng × 4 cột

- Cùng schema với train, **trừ cột `Weekly_Sales`** (đây là biến cần dự báo).
- Khoảng thời gian: **2012-11-02 → 2013-07-26**, nối tiếp ngay sau train (train kết thúc 2012-10-26) → đúng dạng bài toán dự báo tương lai out-of-time, phù hợp TimeSeriesSplit / walk-forward.
- Độ dài horizon test ≈ **39 tuần liên tiếp**.
- **3.169 chuỗi (Store, Dept)** trong test, trong đó **11 cặp Store–Dept chưa từng xuất hiện trong train** → đây là ca **cold-start thật** của chính dataset ở granularity Store-Dept gốc. **Cập nhật 2026-08-19:** sau khi đổi đơn vị dự báo sang (Store, Date), đã xác nhận KHÔNG có Store nào cold-start — cold-start Dept-level ở đây không còn áp dụng cho pipeline hiện tại, xem `docs/00_decisions.md`.

### 2.3. `features.csv` — 8.190 dòng × 12 cột (theo Store, Date — không theo Dept)

| Cột | Missing | Ghi chú |
|---|---|---|
| `Temperature`, `Fuel_Price` | 0 | biến ngoại sinh theo khu vực cửa hàng |
| `MarkDown1..5` | 4.140 – 5.269 / 8.190 (~50–64%) | chương trình khuyến mãi ẩn danh; **chỉ có dữ liệu chủ yếu từ tháng 11/2011 trở đi** — cần kiểm chứng thêm trong EDA, không mặc định fillna(0) |
| `CPI`, `Unemployment` | 585 | chỉ số kinh tế vĩ mô theo khu vực — 585 = đúng 13 tuần cuối test_window (2013-05-03 → 2013-07-26) ở cả 45 Store, do độ trễ công bố thật (đã xác nhận, không phải ngẫu nhiên); xử lý bằng forward-fill theo Store + flag tường minh, xem `docs/00_decisions.md` |
| `IsHoliday` | 0 | trùng với cột trong train/test, dùng để join-check |

### 2.4. `stores.csv` — 45 dòng × 3 cột

`Store`, `Type` (A/B/C — quy mô/loại cửa hàng), `Size` (diện tích, 37.392 – 219.622).

### 2.5. Hệ quả kiến trúc rút ra từ khảo sát data thật

1. **Weekly granularity** → lag/rolling window phải tính theo tuần (vd. lag_1w, lag_4w, rolling_mean_8w), khác hẳn ví dụ "daily lag_1" từng nêu trong slide M5 trước đây.
2. **Weekly_Sales âm** → không thể dùng loss/metric giả định target ≥ 0 một cách mù quáng (vd. MSLE, Poisson loss cần target không âm) — phải xử lý tường minh trong Problem Framing.
3. **MarkDown thiếu theo thời gian (không random)** → đây là dạng "missing not at random" (MNAR) do khuyến mãi chỉ triển khai từ một mốc thời gian; NaN cần được hiểu là "không có chương trình khuyến mãi trong tuần đó", khác với "không có dữ liệu" — cần một cột flag `has_markdown` riêng thay vì chỉ fillna.
4. **11 cặp Store–Dept cold-start thật trong test** → bài toán cold-start không cần mô phỏng, có thể đo trực tiếp bằng cách held-out các cặp tương tự trong validation. *(Cập nhật 2026-08-19: không còn áp dụng ở granularity (Store, Date) hiện tại — xem `docs/00_decisions.md`.)*
5. **3.331 ≠ 45×81** → cần một bước `Data Contract`/schema validation xác nhận danh sách (Store, Dept) hợp lệ, tránh giả định lưới đầy đủ khi tạo lag/rolling (group-by phải theo (Store, Dept) thực tế xuất hiện, không phải cross-join). *(Vẫn áp dụng ở bước ingestion/raw trước aggregate; sau aggregate group-by chỉ còn theo Store.)*
6. **features.csv theo (Store, Date), train theo (Store, Dept, Date)** → bắt buộc phải join, và đây là điểm join dễ sinh lỗi nhân bản dòng / lệch tần suất nếu không kiểm tra kỹ (unit test riêng).

---

## 3. Framing bài toán

**Input:** lịch sử `Weekly_Sales` theo (Store, Dept, Date) + đặc trưng cửa hàng (`Type`, `Size`) + đặc trưng ngoại sinh theo tuần-cửa hàng (`Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment`, `IsHoliday`).

**Output:** `Weekly_Sales` dự báo cho từng (Store, Dept, Date) trong tập test.

### 3 vấn đề domain cốt lõi (rút ra trực tiếp từ data thật, không suy diễn từ nguồn ngoài)

| # | Vấn đề | Bằng chứng từ data | Hướng xử lý (ý tưởng, sẽ tinh chỉnh ở spec) |
|---|---|---|---|
| 1 | **Sales âm** | 1.285 dòng Weekly_Sales < 0 | Không dùng loss/metric giả định không âm mù quáng; cân nhắc giữ nguyên (regression thường) và log rõ trong EDA thay vì clip về 0 một cách vô căn cứ |
| 2 | **Cold-start thật** *(không còn áp dụng ở granularity hiện tại — xem cập nhật 2026-08-19 dưới bảng)* | 11 cặp Store–Dept chỉ có ở test | Feature "có lịch sử hay không" (flag) + fallback theo trung bình Dept/Type thay vì lag cá nhân; đo riêng metric trên nhóm cold-start |
| 3 | **MarkDown MNAR + join phức tạp** | Missing 50–64%, lệch theo Store thay vì Dept | Flag `has_markdown` tường minh; spec riêng cho block join `features.csv` |
| 4 | **Data leakage tiềm ẩn khi tạo lag/rolling & khi join features** | Test nối tiếp train, group-by phải đúng (Store, Dept) thật | Temporal split TRƯỚC feature engineering; unit test kiểm tra "feature tại thời điểm t chỉ dùng dữ liệu ≤ t−1" |
| 5 | **Cấu trúc phân cấp** *(không còn áp dụng — xem cập nhật 2026-08-19 dưới bảng)* | 45 Store × 81 Dept nhưng chỉ 3.331 tổ hợp thật | Global model dùng Store/Dept làm categorical feature (LightGBM native), không train riêng từng chuỗi |
| 6 | **Forecast horizon dài (~39 tuần)** | Test period = 39 tuần liên tiếp | Cần chốt chiến lược recursive vs. direct trước khi thiết kế feature — xem mục 4 |

**Cập nhật 2026-08-19:** đơn vị dự báo đã đổi từ (Store, Dept, Date) sang (Store, Date) — không còn dùng Dept làm categorical feature/entity encoding. Vấn đề #2 (cold-start) và #5 (cấu trúc phân cấp Store×Dept) ở trên mô tả đúng bối cảnh khảo sát ban đầu, nhưng không còn áp dụng cho pipeline hiện tại. Xem `docs/00_decisions.md` "Đổi đơn vị dự báo".

*(Giữ nguyên tinh thần "7 vấn đề domain" từ bản draft trước, nhưng thay các ví dụ minh họa từ M5 bằng bằng chứng đo trực tiếp trên chính dataset của project — loại bỏ hoàn toàn phần "ngoài M5" không cần thiết.)*

---

## 4. Quyết định kiến trúc CHƯA CHỐT — cần team đồng thuận ở Giai đoạn 0

Đây là các lựa chọn nền tảng, cố tình **để mở** trong tài liệu này (theo yêu cầu), team sẽ chốt trong buổi kickoff và ghi quyết định vào `docs/00_decisions.md` (xem CLAUDE.md).

### Lựa chọn A — Direct multi-step, horizon = 39 tuần (bám sát bài toán Kaggle gốc)
- Dự báo toàn bộ 39 tuần test cùng lúc bằng một hoặc nhiều mô hình direct (mỗi mô hình cho một nhóm horizon, vd. h=1-4, h=5-12, h=13-39).
- **Ưu điểm:** khớp đúng bài toán gốc, có thể dùng WMAPE/WMAE có trọng số IsHoliday x5 đúng luật thi Kaggle để so sánh.
- **Nhược điểm:** lag gần nhất (lag_1w) không dùng được cho phần lớn horizon xa → cần thiết kế feature theo từng nhóm horizon, phức tạp hơn.

### Lựa chọn B — Recursive, horizon ngắn (vd. 4–8 tuần), test khoảng đầu của tập test thật
- Chỉ dự báo 4–8 tuần đầu bằng chiến lược recursive (dùng dự báo bước trước làm input bước sau), coi phần còn lại của test.csv là ngoài phạm vi bắt buộc của Module 03.
- **Ưu điểm:** đơn giản hơn, feature engineering tự nhiên hơn (lag_1w luôn có ý nghĩa), phù hợp nếu team ưu tiên chiều sâu phân tích hơn diện rộng.
- **Nhược điểm:** không tận dụng hết test.csv, sai số dồn (error accumulation) theo bước đệ quy cần được đo và báo cáo rõ.

### Lựa chọn C — Hybrid: Direct cho short-horizon (1-4 tuần) + Recursive cho phần còn lại
- Kết hợp để vừa có kết quả khớp toàn bộ test set, vừa giữ được ưu điểm ổn định của direct ở horizon gần.
- **Ưu điểm:** cân bằng giữa độ đầy đủ và độ phức tạp.
- **Nhược điểm:** hai pipeline con cần đồng bộ interface, rủi ro tech debt nếu không thiết kế rõ ranh giới module.

> **Checklist trước khi chốt (bám theo tài liệu "Gợi Ý Chọn Đề Tài Module 03"):** bài toán rõ ràng? dataset đủ? có baseline? metric phù hợp? có error analysis? Team cần trả lời cả 3 lựa chọn trên với checklist này trước khi chọn.

Quyết định cuối cùng (dù chọn A/B/C) đều **không đổi các giai đoạn 1–9 của pipeline** ở tài liệu `02_pipeline_architecture.md` — chỉ ảnh hưởng đến chi tiết bên trong giai đoạn 3 (Feature Engineering) và giai đoạn 5 (Model Training). Đây chính là lý do thiết kế modular theo block độc lập ở mục Nguyên tắc kiến trúc bên dưới.

---

## 4b. Retrain trên train+valid sau khi Optuna tuning xong (định hướng cho Giai đoạn 5-6, chưa bắt buộc chọn ngay)

**Bối cảnh:** Sau khi Optuna đã dùng `train_window`/`valid_window` để tìm `best_params` (giai đoạn 6), một thực hành phổ biến trong ML là fit lại model 1 lần cuối trên `train_window + valid_window` gộp lại (với `best_params` đã cố định) trước khi dự báo `test_window` — tận dụng thêm dữ liệu để model chính xác hơn, vì `valid_window` không còn cần giữ nguyên vẹn cho việc tìm hyperparameter nữa sau khi đã chọn xong.

**Ràng buộc bắt buộc phải tôn trọng nếu áp dụng (không phải tùy chọn, đã chốt ở nơi khác):**

1. **Optuna vẫn chỉ được thấy `train_window`/`valid_window` trong toàn bộ quá trình search** (đã chốt, xem `02_pipeline_architecture.md` mục 3) — việc gộp `train+valid` chỉ xảy ra SAU KHI `best_params` đã chốt, dùng để fit lại đúng 1 lần với tham số cố định, KHÔNG phải một phần của vòng lặp tuning.
2. **Xung đột trực tiếp với thiết kế `calib_window` đã chốt ở `08_uncertainty_conformal.md` mục 3 điểm 3:** hiện tại `calib_window` được lấy từ nửa sau của `valid_window` (nửa đầu dành cho Optuna). Nếu gộp toàn bộ `train_window + valid_window` để retrain, `valid_window` không còn tồn tại độc lập để cung cấp `calib_window` nữa — **residual dùng để hiệu chỉnh conformal interval bắt buộc phải đến từ dữ liệu model retrain chưa từng thấy**, nếu không coverage 95% sẽ bị đánh giá thấp giả tạo (model luôn khớp tốt hơn trên dữ liệu đã fit).
3. Do đó **retrain trên `train+valid` chỉ khả thi nếu team dành riêng một cửa sổ thời gian mới cho `calib_window`** — nằm SAU cả `train_window` lẫn `valid_window` gộp, và trước `test_window`. Với horizon hiện tại (đã chốt: Direct multi-step 39 tuần, `train_window` 2010-02→2011-12, `valid_window` 2012-01→2012-10, `test.csv` thật 2012-11→2013-07), việc này đồng nghĩa phải cắt bớt phần cuối của `valid_window` hiện có để làm `calib_window` mới — KHÔNG dùng lại toàn bộ `valid_window` cho cả retrain lẫn calibration.
4. **Không ảnh hưởng đến khả năng dự báo `test_window` thật** — dù retrain hay không, `test_window` (test.csv, 2012-11→2013-07) vẫn luôn là dữ liệu model chưa từng thấy dưới bất kỳ hình thức nào (không dùng để train, không dùng để tune, không dùng để calibrate).

**Quyết định:** ĐỂ MỞ, chưa chốt — quyết định cụ thể (retrain hay không, và cách chia lại `valid_window`/`calib_window` nếu retrain) sẽ được ghi vào `docs/00_decisions.md` khi Giai đoạn 5-6 (Model Training + Optuna Tuning) thực sự triển khai, cùng lúc với việc chốt cơ chế walk-forward split chi tiết. Không tự ý áp dụng retrain mà không cập nhật lại cách chia `calib_window` — đây là lỗi leakage tiềm ẩn dễ mắc nhất nếu làm không đúng thứ tự.

---

## 5. Nguyên tắc kiến trúc — chống Tech Debt ngay từ thiết kế

1. **Tách logic feature khỏi logic time-boundary.** Không trộn hai việc trong cùng một hàm — nguồn gốc phổ biến nhất của bug leakage khó phát hiện.
2. **Feature theo block bật/tắt độc lập** (Lag / Rolling / Calendar / Store-Dept encoding / MarkDown-Promo / Macro). Ablation-test không đòi hỏi viết lại pipeline.
3. **Một evaluation layer dùng chung cho mọi model** — Baseline (Naive) → Decision Tree → ... → LightGBM → XGBoost đều so sánh công bằng, cùng một cách tính WMAE/WMAPE.
4. **Optuna tái dùng đúng cơ chế walk-forward của Evaluation** — không viết quy trình đánh giá riêng chỉ cho tuning.
5. **Ghi lại quyết định thiết kế từ Giai đoạn 0** vào `docs/00_decisions.md` — đơn vị dự báo, horizon, chiến lược recursive/direct/hybrid — để team không tranh cãi lại giữa chừng dự án.
6. **Mọi giả định quan trọng đều có test case tương ứng** (xem `05_test_plan.md`) — không có giả định "ngầm hiểu".

---

## 6. Metric đề xuất

- **Chính:** WMAPE hoặc WMAE có trọng số theo `IsHoliday` (x5 cho tuần lễ, đúng luật đánh giá gốc của cuộc thi Kaggle) — cho phép so sánh trực tiếp với leaderboard công khai nếu cần.
- **Phụ:** MAE, RMSE tổng quát; tách riêng theo nhóm fast/slow-moving Dept, theo Store Type (A/B/C), theo cold-start vs. có lịch sử, theo tuần lễ vs. không lễ.
- **Không dùng riêng Accuracy/R² làm tiêu chí chính** vì đây là bài toán regression có phân phối lệch (đuôi dài, giá trị âm hiếm).

---

## 9. Bước tiếp theo

1. Team họp chốt Lựa chọn A/B/C ở mục 4 → ghi vào `docs/00_decisions.md`.
2. Setup Temporal Split + Data Validation (giai đoạn 1–2 pipeline) trước khi ai chạm vào feature engineering.
3. Viết spec kỹ thuật ngắn cho từng block feature (input/output/ranh giới thời gian) — xem `03_data_io_diagram.md`.
4. Thiết kế và chạy bộ unit test chống leakage trước khi model đầu tiên được train — xem `05_test_plan.md`.
5. Setup môi trường venv/pip theo `06_environment_setup.md`, xác thực bằng `scripts/check_env.py`.
6. Sau khi có model điểm đầu tiên (Giai đoạn 4-6) → triển khai Split Conformal Prediction theo `08_uncertainty_conformal.md`.
7. Sau khi có đủ metric/predictions/SHAP → dựng dashboard theo `07_dashboard_spec.md`.
