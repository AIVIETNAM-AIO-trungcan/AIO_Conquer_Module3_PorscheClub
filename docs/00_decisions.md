# Log quyết định kiến trúc — Sales & Demand Forecasting

**AIO Conquer 2026 — Module 03 · Project 3.2**

> File này được cập nhật mỗi khi team chốt một quyết định kiến trúc quan trọng. Không xóa quyết định cũ — nếu thay đổi, thêm mục mới và ghi rõ lý do thay đổi, giữ lại lịch sử để tránh tranh cãi lặp lại.

---

## Trạng thái: **ĐÃ CHỐT TOÀN BỘ** — các quyết định nền tảng (đơn vị dự báo, horizon, chiến lược) đã chốt ngày 2026-08-18

Các quyết định về horizon/chiến lược/xử lý dữ liệu đã được điền sau buổi họp Giai đoạn 0 (Problem Framing), tham chiếu `docs/01_ideation.md` mục 4. Các quyết định kỹ thuật bổ sung (conformal, CatBoost, môi trường, dashboard) đã được chốt theo yêu cầu ngày 2026-08-18.

| Hạng mục | Lựa chọn đã chốt | Ngày chốt | Người quyết định | Lý do |
|---|---|---|---|---|
| Đơn vị dự báo | `(Store, Dept, Date)`, tần suất tuần (weekly) | 2026-08-18 | DTC | Khớp đúng granularity thật của `train.csv`/`test.csv`, xem `01_ideation.md` mục 2.1 |
| Forecast horizon | Lựa chọn A — Direct multi-step, horizon = 39 tuần, chia 3 nhóm horizon: h=1-4, h=5-12, h=13-39 | 2026-08-18 | DTC | Khớp đúng bài toán Kaggle gốc (test.csv dài 39 tuần liên tiếp), cho phép dùng WMAE/WMAPE chuẩn để đối chiếu leaderboard; đánh đổi là feature engineering phức tạp hơn theo từng nhóm horizon vì lag gần không dùng được cho horizon xa — xem `01_ideation.md` mục 4 Lựa chọn A |
| Chiến lược (recursive/direct/hybrid) | Direct multi-step (không recursive) | 2026-08-18 | DTC | Đi kèm quyết định horizon ở trên; tránh sai số dồn (error accumulation) của recursive trên horizon dài 39 tuần |
| Metric chính | **Cả WMAE và WMAPE**, cùng trọng số IsHoliday x5 — báo cáo song song trong mọi bảng metric | 2026-08-18 | DTC | WMAE khớp đúng luật đánh giá gốc Kaggle (đối chiếu leaderboard); WMAPE chuẩn hóa theo %, dễ so sánh giữa các Dept có quy mô doanh số khác nhau — không đánh đổi lẫn nhau nên báo cáo cả hai thay vì chọn một, xem `01_ideation.md` mục 6 |
| Xử lý Weekly_Sales âm | Giữ nguyên, không clip về 0 | 2026-08-18 | DTC | 1.285 dòng âm là tín hiệu thật (trả hàng/điều chỉnh sổ sách), không phải nhiễu; không dùng loss/metric giả định target ≥ 0 (loại MSLE, Poisson loss) — xem `01_ideation.md` mục 2.5 và mục 3 vấn đề #1 |
| Xử lý MarkDown missing | Flag `has_markdown` tường minh (đã chốt — xem `01_ideation.md` mục 2.5) | 2026-08-18 | DTC | Missing 50-64% là MNAR (khuyến mãi chỉ triển khai từ một mốc thời gian), NaN nghĩa là "không có khuyến mãi" chứ không phải "thiếu dữ liệu" — fillna(0) mù quáng sẽ đánh mất phân biệt này |
| Xử lý cold-start (11 cặp Store-Dept) | Flag `has_history=False` (bool) là feature bắt buộc; fallback prediction dùng giá trị trung bình theo Dept hoặc Store Type khi không có lịch sử lag cá nhân; đo riêng metric nhóm cold-start ở Giai đoạn 7 (Evaluation) | 2026-08-18 | DTC | 11 cặp Store-Dept trong test.csv chưa từng xuất hiện ở train là cold-start thật của chính dataset; không được để lag NaN âm thầm lọt vào model — xem `01_ideation.md` mục 2.5 điểm 4 và mục 3 vấn đề #2 |
| Phương pháp khoảng tin cậy 95% | Split Conformal Prediction (đã chốt — xem `08_uncertainty_conformal.md`) | 2026-08-18 | DTC | Áp dụng đồng loạt mọi model cây, không cần train lại theo quantile loss |
| Tỷ lệ tách `calib_window` từ `valid_window` | 50/50 với phần dùng cho Optuna | 2026-08-18 | DTC | Cân bằng giữa đủ dữ liệu cho Optuna tìm best trial và đủ mẫu để calibration cho coverage 95% ổn định — xem `08_uncertainty_conformal.md` mục 3 |
| Model tree-based bổ sung (hướng nâng cao) | Loại bỏ, không dùng CatBoost (đảo ngược quyết định trước — xem mục "Loại bỏ CatBoost" bên dưới) | 2026-08-18 | DTC | Thu hẹp phạm vi về đúng 5 model bắt buộc của Module 03 |
| Công cụ quản lý môi trường | venv + pip + `pyproject.toml` (đã chốt — xem `06_environment_setup.md`) | 2026-08-18 | DTC | Không cần cài Anaconda/Docker, wheel sẵn cho LightGBM/XGBoost trên Python 3.10–3.12 |
| Dạng dashboard kết quả | Streamlit app (đã chốt — xem `07_dashboard_spec.md`) | 2026-08-18 | DTC | Tích hợp trực tiếp Python, mentor chạy 1 lệnh `streamlit run` |
| Run tracking/versioning output | Tự chế run_id + thư mục `runs/<run_id>/` + pointer `latest_run.txt` + `run_history.csv` (đã chốt — xem mục dưới) | 2026-08-19 | DTC | Tránh ghi đè output mỗi lần chạy pipeline, giữ lịch sử để so sánh cải thiện; không dùng MLflow/DVC để tránh dependency nặng |
| Xử lý CPI/Unemployment missing (585 dòng đuôi test_window) | Forward-fill theo từng Store + flag tường minh `cpi_is_forward_filled`/`unemployment_is_forward_filled` (đã chốt — xem mục dưới) | 2026-08-19 | DTC | 585 dòng = đúng 13 tuần cuối test_window ở cả 45 Store, do độ trễ công bố macro index thật, không phải ngẫu nhiên; forward-fill hợp lý hơn fillna(0)/mean vì CPI/Unemployment biến động rất chậm theo tháng |
| Đơn vị dự báo (CẬP NHẬT 2026-08-19) | `(Store, Date)` — GHI ĐÈ quyết định `(Store, Dept, Date)` chốt 2026-08-18, xem mục chi tiết bên dưới | 2026-08-19 | DTC | Giảm độ phức tạp mô hình hóa (81 Dept/Store quá nhiều, nhiều chuỗi ngắn/thưa) + mục tiêu dashboard/báo cáo tập trung cấp Store dễ diễn giải hơn; hệ quả: không còn tạo được `submission.csv` đúng format Kaggle gốc |
| Buffer nối train_window/valid_window cho Lag/Rolling/Macro | Gộp bảng (train + phần chưa biết target) trước khi tính lag/rolling/macro, tách lại sau — đã implement, chính thức hóa thành quy tắc kiến trúc (xem mục dưới) | 2026-08-19 | DTC | Loại bỏ NaN giả tạo ở các dòng đầu mỗi tập; đúng sơ đồ kiến trúc gốc; đã kiểm chứng bằng thực nghiệm không leakage |

---

## [2026-08-18] Forecast horizon & chiến lược dự báo

**Bối cảnh:** `test.csv` có 39 tuần liên tiếp cần dự báo cho mỗi (Store, Dept). `01_ideation.md` mục 4 để mở 3 lựa chọn A/B/C, cố tình chưa chốt để team quyết định trong buổi kickoff.

**Các lựa chọn đã xem xét:**
- A. Direct multi-step, horizon = 39 tuần, chia nhóm horizon
- B. Recursive, horizon ngắn 4-8 tuần
- C. Hybrid — direct cho short-horizon + recursive cho phần còn lại

**Quyết định:** Lựa chọn A — Direct multi-step, dự báo toàn bộ 39 tuần test, chia thành 3 nhóm horizon: h=1-4, h=5-12, h=13-39. Mỗi nhóm horizon có thể dùng feature set/model riêng.

**Lý do:** Khớp đúng bài toán Kaggle gốc, cho phép dùng WMAE/WMAPE chuẩn để so sánh với leaderboard công khai nếu cần. Chấp nhận đánh đổi: feature engineering phức tạp hơn vì lag_1w không có ý nghĩa cho horizon xa (h=13-39) — cần thiết kế feature riêng theo từng nhóm horizon.

**Ảnh hưởng tới các module:** `src/sales_forecast/features/` (feature phải phân biệt theo nhóm horizon), `src/sales_forecast/models/` (có thể cần train riêng theo nhóm horizon), `pipelines/run_train_full.py`. Không ảnh hưởng cấu trúc 10 giai đoạn ở `02_pipeline_architecture.md`.

**Test case liên quan:** Cần bổ sung khi bắt đầu code Giai đoạn 3/5 — ví dụ test đảm bảo feature lag ngắn hạn không bị dùng nhầm cho nhóm horizon xa.

---

## [2026-08-18] Metric chính — báo cáo song song WMAE và WMAPE

**Bối cảnh:** `01_ideation.md` mục 6 đề xuất WMAE có trọng số IsHoliday x5 làm metric chính (đúng luật Kaggle gốc), nhưng không loại trừ WMAPE.

**Các lựa chọn đã xem xét:** Chỉ WMAE / Chỉ WMAPE / Cả hai song song.

**Quyết định:** Báo cáo **cả WMAE và WMAPE**, cùng trọng số IsHoliday x5, ở mọi bảng metric (Evaluation Layer, Dashboard, báo cáo kỹ thuật) — không chọn một làm "chính" duy nhất.

**Lý do:** WMAE khớp luật đánh giá gốc Kaggle (đối chiếu leaderboard công khai). WMAPE chuẩn hóa theo %, dễ diễn giải khi so sánh giữa các Dept có quy mô doanh số chênh lệch lớn. Hai metric bổ trợ nhau, không đánh đổi lẫn nhau.

**Ảnh hưởng tới các module:** `src/sales_forecast/evaluation/metrics.py` (implement cả hai hàm `weighted_mae` và `weighted_mape`), mọi bảng report ở `reports/metrics/` và tab "Model Comparison" của dashboard (`docs/07_dashboard_spec.md`) phải hiển thị cả 2 cột.

**Test case liên quan:** `tests/test_evaluation/test_metrics_correctness.py` (đã có skeleton A11 ở `05_test_plan.md`) — cần mở rộng để test cả `weighted_mae` và `weighted_mape`.

---

## [2026-08-18] Xử lý cold-start (11 cặp Store-Dept)

**Bối cảnh:** 11 cặp (Store, Dept) trong `test.csv` chưa từng xuất hiện trong `train.csv` — cold-start thật của dataset, không phải mô phỏng.

**Các lựa chọn đã xem xét:**
- Flag `has_history` + fallback theo Dept/Type trung bình
- Loại khỏi phạm vi đánh giá chính, chỉ ghi chú hạn chế

**Quyết định:** Thêm cột `has_history` (bool) là feature bắt buộc trong `feature_matrix`. Khi `has_history=False`, dùng giá trị trung bình theo Dept hoặc Store Type làm fallback thay vì lag cá nhân (vốn sẽ là NaN). Đo riêng metric (WMAE/WMAPE) cho nhóm cold-start trong Giai đoạn 7 (Evaluation & Error Analysis).

**Lý do:** Đây là cold-start thật của chính dataset (không cần mượn ví dụ ngoài), nên phải đo trực tiếp thay vì né tránh — phù hợp với checklist "có error analysis" theo tài liệu "Gợi Ý Chọn Đề Tài Module 03".

**Ảnh hưởng tới các module:** `src/sales_forecast/features/pipeline.py` (sinh cột `has_history`), `src/sales_forecast/evaluation/error_analysis.py` (breakdown theo cold-start vs. có lịch sử — đã có trong `02_pipeline_architecture.md` mục 4 S7d).

**Test case liên quan:** `tests/test_features/test_cold_start_handling.py` (đã có skeleton A8 ở `05_test_plan.md` mục 2.4).

---

## [2026-08-18] WMAPE bùng nổ do Weekly_Sales gần 0 — giữ nguyên công thức, không loại trừ dòng

**Bối cảnh:** Chạy thử baseline (`pipelines/run_train_baseline.py`) trên `data/raw/` thật phát hiện WMAPE trên `valid_window` (2012) đạt giá trị bất thường (~96 triệu %), trong khi WMAE vẫn hợp lý (~2,218). Điều tra cho thấy 345 dòng trong `valid_window` có `|Weekly_Sales| < 1.0` (nhỏ nhất 0.01, ví dụ Store 1 Dept 99), khiến `APE = |y_true-y_pred|/|y_true|` ra giá trị cực lớn dù sai số tuyệt đối nhỏ, áp đảo trung bình có trọng số trên toàn bộ ~127K dòng.

**Các lựa chọn đã xem xét:**
- Giữ nguyên công thức WMAPE, ghi chú rõ hạn chế trong docstring/output
- Tăng `epsilon`/denominator floor (vd. `max(|y_true|, 1.0)`) để chặn outlier
- Loại trừ các dòng `|y_true|` quá nhỏ khi tính riêng WMAPE

**Quyết định:** Giữ nguyên công thức WMAPE chuẩn (`epsilon=1e-8` chỉ để tránh chia đúng 0), KHÔNG loại trừ hay áp floor tùy tiện lên các dòng sales gần 0. WMAE là metric ổn định chính để đối chiếu; WMAPE luôn được báo cáo kèm WMAE, không dùng riêng lẻ. Diễn giải/phân tích chi tiết nhóm sales gần 0 để dành cho Giai đoạn 7 (Evaluation & Error Analysis).

**Lý do:** Nhất quán với quyết định "giữ nguyên Weekly_Sales âm/gần 0, không xử lý mù quáng" đã chốt trước đó — áp floor hay loại trừ dòng sẽ là một dạng che giấu tín hiệu thật tương tự việc clip sales âm về 0, vi phạm tinh thần đã thống nhất. Đây là hạn chế toán học đã biết của MAPE với mẫu số gần 0, không phải lỗi implementation.

**Ảnh hưởng tới các module:** `src/sales_forecast/evaluation/metrics.py` (docstring `weighted_mape` ghi rõ giới hạn), `pipelines/run_train_baseline.py` (in cảnh báo số dòng `|Weekly_Sales| < 1.0` mỗi lần chạy). Giai đoạn 7 (Evaluation & Error Analysis) khi triển khai cần thêm 1 lát cắt riêng theo "sales gần 0 vs. sales bình thường".

**Test case liên quan:** `tests/test_evaluation/test_metrics_correctness.py::test_weighted_mape_handles_zero_sales_without_inf` (đảm bảo không trả về inf/NaN, không đảm bảo giá trị nhỏ).

---

## [2026-08-18] Loại bỏ CatBoost khỏi danh sách model

**Bối cảnh:** Mục "Model tree-based bổ sung (hướng nâng cao)" ở bảng tóm tắt phía trên đã chốt dùng CatBoost cùng ngày 2026-08-18. Sau đó nhận yêu cầu loại bỏ CatBoost khỏi thuật toán.

**Quyết định:** Hủy quyết định dùng CatBoost. Project chỉ giữ đúng 5 model bắt buộc theo Module 03: Decision Tree, Random Forest, AdaBoost, Gradient Boost, LightGBM/XGBoost. Không có model tree-based bổ sung nào ở hướng nâng cao tại thời điểm này.

**Lý do:** Theo yêu cầu trực tiếp, không nêu lý do nghiệp vụ cụ thể khác ngoài việc thu hẹp phạm vi về đúng 5 model bắt buộc.

**Ảnh hưởng tới các module:** `src/sales_forecast/models/` (bỏ wrapper CatBoost nếu có), `configs/optuna.yaml` (bỏ search space CatBoost), `pyproject.toml` (bỏ dependency `catboost`), `tests/test_models/test_model_interface_consistency.py` (bỏ CatBoost khỏi danh sách model được test), `docs/01_ideation.md` mục 8, `docs/06_environment_setup.md`, `docs/08_uncertainty_conformal.md`, `README.md`, `scripts/check_env.py`.

**Test case liên quan:** Không cần test case mới; loại bỏ tham chiếu CatBoost khỏi test hiện có.

---

## [2026-08-19] Run tracking/versioning cho output pipeline

**Bối cảnh:** Mỗi lần chạy pipeline (`pipelines/run_train_baseline.py`, và các
pipeline tương lai `run_train_full.py`, `run_optuna_tuning.py`,
`run_conformal_calibration.py`, `run_evaluation.py`, `run_shap.py`,
`run_end_to_end.py`) ghi output ra `reports/` và `data/predictions/` theo path cố
định (vd. `reports/figures/decision_tree.png`). File cũ bị ghi đè hoàn toàn ở lần
chạy sau — không thể so sánh WMAE/WMAPE/coverage giữa các lần chạy để làm bằng
chứng cải thiện (vd. trước/sau thêm feature, trước/sau Optuna tuning). Khảo sát
repo tại thời điểm này cho thấy chỉ `run_train_baseline.py` đã chạy thật; các
pipeline khác và `app/dashboard.py` chưa được implement — đây là thời điểm tốt
nhất để thiết kế versioning trước khi các module ghi output còn lại được viết.

**Các lựa chọn đã xem xét:**
- Tự chế cơ chế run_id + thư mục con theo run (không dependency mới)
- MLflow (tracking server/UI riêng, cần thêm dependency khá nặng)
- DVC (data version control qua git + remote storage — repo hiện chưa có git)

**Quyết định:** Tự chế cơ chế run tracking, không dùng MLflow/DVC. Chi tiết:
- `run_id` định dạng `{pipeline_name}_{YYYYMMDD_HHMMSS}` (vd.
  `train_baseline_20260819_143205`), thêm suffix `_2`/`_3`... nếu trùng (2 lần chạy
  cùng giây, cùng pipeline) — không bao giờ ghi đè âm thầm.
- Output ghi vào `reports/runs/<run_id>/` và `data/predictions/runs/<run_id>/` thay
  vì path cố định.
- Pointer file `latest_run.txt` (KHÔNG dùng symlink vì Windows cần quyền
  admin/Developer Mode) — 2 bản độc lập cho `reports/` và `data/predictions/`, chỉ
  cập nhật bởi run thực sự có ghi file vào base_dir đó, và chỉ cập nhật khi run
  thành công (`status="success"`) để run lỗi/dở dang không bao giờ làm dashboard
  đọc nhầm dữ liệu thiếu.
- `reports/run_history.csv` (append-only, long format 1 dòng/model/run) lưu
  `run_id, started_at, pipeline_name, model_name, wmae, wmape, coverage_95, status`
  — tra cứu nhanh lịch sử cải thiện qua thời gian mà không cần mở từng thư mục run.
- `manifest.json` mỗi run lưu config snapshot (nội dung các file `configs/*.yaml`
  đã dùng), `metrics_summary`, `status`, `started_at`/`finished_at`.
- Ghi atomic (file tạm + `os.replace()`) cho mọi file quan trọng để tránh
  half-written nếu tiến trình bị ngắt giữa chừng.

**Lý do:** Tránh thêm dependency nặng (MLflow kéo theo alembic/sqlalchemy...),
giữ đúng triết lý venv+pip đơn giản đã chốt ở `docs/06_environment_setup.md`. Giải
pháp tự chế tích hợp tự nhiên với dashboard Streamlit đã thiết kế (chỉ là đọc thêm
file JSON/CSV/pointer qua `app/data_loader.py`, đúng nguyên tắc "data_loader.py là
nơi duy nhất đọc file") — dùng MLflow sẽ tạo 1 UI riêng (`mlflow ui`) trùng lặp
chức năng với dashboard, gây nhầm lẫn "xem kết quả ở đâu". DVC không phù hợp vì
repo hiện chưa có git và mục tiêu là so sánh metric theo thời gian, không phải
content-addressable dedup của data lớn.

**Ảnh hưởng tới các module:** `src/sales_forecast/utils/io.py` (mới — ghi/đọc
atomic dùng chung), `src/sales_forecast/utils/run_tracking.py` (mới —
`RunContext`, `start_run`, `get_latest_run_id`, `list_runs`),
`pipelines/run_train_baseline.py` (đã tích hợp `RunContext`, path
`decision_tree.png` giờ qua `run_ctx.reports_path(...)`), mọi pipeline tương lai
(`run_train_full.py`, `run_optuna_tuning.py`, `run_conformal_calibration.py`,
`run_evaluation.py`, `run_shap.py`, `run_end_to_end.py` — khi được viết phải dùng
chung `start_run()`/`RunContext`, `run_end_to_end.py` dùng đúng 1 `run_id` cho toàn
bộ chuỗi con), `app/data_loader.py` (khi được viết — mọi hàm đọc nhận tham số
`run_id: str | None = None`, resolve qua `get_latest_run_id()` khi `None`).

**Test case liên quan:** `tests/test_utils/test_run_tracking.py` — giả định A25-A33
(xem `docs/05_test_plan.md`), toàn bộ đã pass. `tests/test_app/test_data_loader.py`
— giả định A34-A35, sẽ viết khi `app/data_loader.py` được implement.

---

## [2026-08-19] Xử lý CPI/Unemployment missing ở đuôi test_window (585 dòng)

**Bối cảnh:** Khảo sát trực tiếp `data/raw/features.csv` xác nhận CPI/Unemployment thiếu đúng 585 dòng = 13 tuần cuối test_window (2013-05-03 → 2013-07-26) ở TẤT CẢ 45 Store, pattern giống hệt nhau giữa 2 cột — do độ trễ công bố macro index thật, không phải lỗi ngẫu nhiên. Temperature/Fuel_Price/MarkDown1-5 phủ đủ toàn bộ test_window, không có vấn đề tương tự. Phát hiện này cũng làm rõ một điểm quan trọng khác: `features.csv` nói chung phủ hết test_window, tức biến ngoại sinh là dữ liệu quan sát được tại thời điểm dự báo (given), không cần tự dự báo — xem cập nhật tương ứng ở `docs/03_data_io_diagram.md` mục 3.

**Các lựa chọn đã xem xét:**
- `fillna(0)` hoặc `fillna(mean toàn cục)` — loại vì tạo bước nhảy giả tạo, sai lệch với đặc tính biến động rất chậm theo tháng của CPI/Unemployment
- Forward-fill từ giá trị công bố gần nhất theo từng Store, kèm flag tường minh
- Bỏ CPI/Unemployment khỏi feature set — loại vì mất thông tin không cần thiết (chỉ 13/39 tuần cuối bị ảnh hưởng)

**Quyết định:** Forward-fill CPI/Unemployment theo từng Store (dùng giá trị công bố gần nhất trước đó theo thời gian, không nhìn tương lai), kèm flag tường minh `cpi_is_forward_filled`/`unemployment_is_forward_filled` (bool), nhất quán với triết lý `has_markdown` đã chốt — không fillna mù quáng.

**Lý do:** CPI/Unemployment là chỉ số kinh tế vĩ mô biến động rất chậm theo tháng/quý, forward-fill hợp lý hơn `fillna(0)` (sai đơn vị/thang đo) hoặc `fillna(mean)` (che giấu xu hướng thật). Flag tường minh cho phép model/error analysis phân biệt dòng có giá trị công bố thật vs. dòng suy ra từ độ trễ, đúng nguyên tắc "NaN có ý nghĩa nghiệp vụ khác nhau" (CLAUDE.md mục 4 rule 9). Không hard-code cửa sổ "13 tuần" trong code — `ffill` tự nhiên chỉ điền đúng chỗ có NaN.

**Ảnh hưởng tới các module:** `src/sales_forecast/features/macro.py` (hàm `_add_macro_forward_fill`, gọi trong `add_macro_features`), `docs/03_data_io_diagram.md` (mục 1 ERD, mục 3 bảng Feature Matrix — thêm 2 cột flag), `docs/01_ideation.md` mục 2.3.

**Test case liên quan:** `tests/test_features/test_macro_forward_fill.py` (mới, 4 test đã pass) — xem `docs/05_test_plan.md` giả định A36-A38.

---

## [2026-08-19] Đổi đơn vị dự báo: (Store, Dept, Date) → (Store, Date) — bỏ Dept

**Bối cảnh:** Quyết định ngày 2026-08-18 đã chốt đơn vị dự báo `(Store, Dept, Date)`. Sau buổi thảo luận nhóm tiếp theo, khảo sát trực tiếp trên `data/raw/` xác nhận: mỗi (Store, Date) trong train có đầy đủ 45 Store × mọi tuần (6435 combos = 45 × 143 tuần, khớp chính xác); `IsHoliday` nhất quán 100% trong mọi (Store, Date); **KHÔNG có Store nào cold-start ở test** (khác hẳn cold-start 11 cặp Store-Dept ở granularity cũ, cả 45 Store đều có trong train); sau khi aggregate SUM Weekly_Sales theo Dept thì không còn dòng nào âm (min = 209,986, do trung bình ~65 Dept/Store-week cộng lại lấn át các dòng âm lẻ tẻ). Ngoài ra, `docs/progress_week1_report.md` mục 3.2 đã ghi nhận bằng chứng nội bộ: baseline Decision Tree thua Naive baseline vì ordinal encoding của ~2.000+ tổ hợp Store-Dept không đủ để một cây đơn tái tạo hành vi tra cứu chính xác — càng củng cố lý do giảm về 45 Store.

**Các lựa chọn đã xem xét:**
- Giữ nguyên `(Store, Dept, Date)` như đã chốt 2026-08-18
- Đổi sang `(Store, Date)`, aggregate SUM Weekly_Sales theo Dept

**Quyết định:** Đổi đơn vị dự báo sang `(Store, Date)`. Aggregate SUM Weekly_Sales theo (Store, Date) ngay sau Giai đoạn 1 (Ingestion & Validation), trước Giai đoạn 2 (Temporal Split) — hàm `aggregate_to_store_date()` trong `src/sales_forecast/ingestion/loaders.py`. Đây là quyết định GHI ĐÈ quyết định "Đơn vị dự báo" chốt ngày 2026-08-18 — quyết định cũ được GIỮ NGUYÊN trong log này (không xóa), chỉ không còn hiệu lực áp dụng từ 2026-08-19 trở đi.

**Lý do:** (1) Giảm độ phức tạp mô hình hóa — 81 Dept/Store tạo ra ~2.000+ chuỗi ngắn/thưa, chi phí tính toán/tuning lớn hơn đáng kể so với 45 chuỗi Store; bằng chứng thực nghiệm ở `progress_week1_report.md` mục 3.2 cho thấy ordinal encoding của quá nhiều tổ hợp Store-Dept làm hại chất lượng model cây đơn giản. (2) Mục tiêu dashboard/báo cáo tập trung cấp Store dễ diễn giải hơn cho người xem cuối (mentor/giảng viên), phù hợp yêu cầu cốt lõi Module 03 là thể hiện quy trình ML hoàn chỉnh và khả năng giải thích kết quả hơn là tối đa hóa độ chi tiết granularity.

**Hệ quả quan trọng:** KHÔNG còn tạo được `submission.csv` đúng format Kaggle gốc (yêu cầu Dept trong id `Store_Dept_Date`). Project từ thời điểm này không còn mục tiêu đối chiếu Kaggle leaderboard công khai (ghi đè tinh thần dòng "Metric chính" ở bảng tóm tắt, vốn nhắc tới "đối chiếu leaderboard" — không xóa quyết định đó, chỉ không còn áp dụng được phần đối chiếu leaderboard). Đối tượng bàn giao là dashboard Streamlit + báo cáo kỹ thuật nội bộ, đúng yêu cầu cốt lõi của Module 03. Cold-start (11 cặp Store-Dept, quyết định 2026-08-18) không còn áp dụng — đã xác nhận không có Store cold-start ở granularity mới; `has_history` giữ lại trong code/schema cho mục đích tổng quát hóa/phòng thủ nhưng dự kiến luôn `True` trên dữ liệu hiện tại. Cột `is_cold_start` và `product_age_weeks` đã bị loại bỏ khỏi đặc tả `predictions_long`/Feature Matrix (xem `docs/03_data_io_diagram.md`).

**Ảnh hưởng tới các module:** `src/sales_forecast/ingestion/schema.py` (thêm `train_aggregated_schema`/`test_aggregated_schema`, schema raw gốc vẫn giữ Dept vì validate dữ liệu CSV gốc trước aggregate), `src/sales_forecast/ingestion/loaders.py` (hàm mới `aggregate_to_store_date`), `src/sales_forecast/ingestion/validators.py` (`validate_train_aggregated_schema`/`validate_test_aggregated_schema`), `src/sales_forecast/features/pipeline.py` (`_add_has_history`, `build_feature_matrix` group_cols, thêm `load_enabled_blocks_from_config`), `src/sales_forecast/features/store_encoding.py` (đổi tên từ `store_dept_encoding.py`, hàm `encode_store` thay `encode_store_dept`), `src/sales_forecast/models/baseline.py` (`NaiveSameWeekLastYear` groupby/index), `pipelines/run_train_baseline.py` (thêm bước 1b Aggregate, MultiIndex reindex bỏ Dept, đọc `enabled_blocks` từ `configs/features.yaml`), `configs/data.yaml` (bỏ `dept_column`), `configs/features.yaml` (đổi key block `store_dept_encoding` → `store_encoding`), `docs/01_ideation.md`/`02_pipeline_architecture.md`/`03_data_io_diagram.md`/`07_dashboard_spec.md` (rà soát, chú thích không còn áp dụng — không xóa số liệu khảo sát gốc), `docs/progress_week1_report.md` (ghi chú số liệu WMAE baseline cũ không còn so sánh trực tiếp được).

**Test case liên quan:** `tests/test_ingestion/test_aggregate_to_store_date.py` (mới, A39-A41), `tests/test_ingestion/test_schema_validation.py` (mở rộng), `tests/test_ingestion/test_join_integrity.py` (thêm `test_features_join_after_aggregate_is_one_to_one`), `tests/test_features/test_lag_rolling_no_future_leak.py` (sửa, A42), `tests/test_features/test_cold_start_handling.py` (sửa, A43), `tests/test_features/test_feature_config_loading.py` (mới — đọc `features.yaml` runtime), `tests/test_models/test_model_interface_consistency.py` (sửa, A44), `tests/test_splitting/test_temporal_split_no_leakage.py` (sửa) — xem `docs/05_test_plan.md` giả định A39-A44.

---

## [2026-08-19] Chính thức hóa cơ chế buffer nối train_window/valid_window cho Lag/Rolling/Macro

**Bối cảnh:** Sau buổi thảo luận nhóm, team đề xuất gộp train + test + features thành 1 bảng, tính rolling/lag cho biến ngoại sinh trên bảng gộp, sau đó tách lại train/validate/test, rồi tính rolling/lag cho biến target — validate tách nhãn riêng để đánh giá. Mối lo cụ thể: NaN ở các giá trị biên nối giữa train/validation/test.

Kiểm định bằng code + thực nghiệm thật xác nhận: `train_window` và `valid_window` nối liền nhau về thời gian (gap đúng 1 tuần, đo trực tiếp trên `data/raw/`: `train_w` kết thúc 2011-12-30, `valid_w` bắt đầu 2012-01-06; tương tự `valid_w` kết thúc 2012-10-26 nối liền `test.csv` thật bắt đầu 2012-11-02). `build_feature_matrix` (`src/sales_forecast/features/pipeline.py`) hiện tại **đã** `pd.concat([train_df, test_df])` (tham số `test_df` hiện đang nhận `valid_window` đã bỏ cột target) rồi mới gọi `add_lag_features`/`add_rolling_features` trên bảng gộp — đây chính là cơ chế buffer window nhóm đề xuất, đã tồn tại cho ranh giới train↔valid nhưng chưa được ghi nhận tường minh thành quyết định kiến trúc. `docs/02_pipeline_architecture.md` mục 2 đã vẽ sẵn sơ đồ `D1(train) & D2(valid) & D3(test) → E(Feature Engineering)`, đúng hướng đề xuất.

**Các lựa chọn đã xem xét:**
- Tính lag/rolling riêng lẻ trên từng tập (train, valid, test) — bị loại vì tạo NaN giả tạo ở các dòng đầu mỗi tập, dù thực tế có đủ lịch sử thật từ tập liền trước
- Gộp bảng (train + phần chưa biết target) trước khi tính lag/rolling/macro, tách lại sau — đã chọn, đã implement từ trước cho ranh giới train↔valid

**Quyết định:** Chính thức hóa cơ chế gộp bảng trước khi tính Lag/Rolling/Macro làm quy tắc kiến trúc chuẩn của Giai đoạn 3 (`build_feature_matrix`). Hiện áp dụng cho ranh giới train↔valid (đã implement, không cần sửa code). Ranh giới valid↔test.csv thật sẽ áp dụng cùng nguyên tắc khi Giai đoạn 5 thực sự cần dự báo `test_window` — không mở rộng code ngay bây giờ, tránh sửa cho nhu cầu chưa dùng tới.

**Xác nhận đây KHÔNG phải leakage — cơ sở toán học:** `add_lag_features` dùng `merge` dịch ngày theo group (chỉ khớp đúng offset quá khứ `t-n`); `add_rolling_features` dùng `.shift(1).rolling(...)` (luôn dịch 1 bước trước khi rolling). Cả 2 hàm không bao giờ nhìn dòng có Date > dòng hiện tại, bất kể ranh giới train/valid/test nằm ở đâu trong bảng gộp. Thực nghiệm xác nhận: mô phỏng chuỗi 6 tuần với 3 tuần cuối target=NaN, dòng đầu tiên sau ranh giới nhận đúng lag/rolling từ quá khứ thật (không NaN oan); các dòng sau đó tự nhiên NaN dần vì valid/test chưa có target thật để tự lag cho chính nó — đây là **giới hạn tự nhiên của direct multi-step forecasting** (đã chốt ở quyết định horizon/strategy 2026-08-18), không phải bug.

**Phân biệt quan trọng giữa 2 nhóm biến:** nhóm **macro** (Temperature/Fuel_Price/CPI/Unemployment/MarkDown) an toàn khi tính rolling/lag trên toàn bộ chuỗi gộp train+valid+test vì là given data (đã xác nhận `features.csv` phủ hết đến 2013-07-26, xem quyết định "Xử lý CPI/Unemployment missing" ở trên). Nhóm **target** (Weekly_Sales) an toàn khi gộp bảng để tránh NaN biên, nhưng valid/test không thể tự cung cấp lag cho chính nó ở các bước xa hơn 1 — đây là giới hạn tự nhiên, không phải thiếu sót cần vá.

**Lý do:** Loại bỏ NaN giả tạo ở các dòng đầu mỗi tập; đúng với sơ đồ kiến trúc gốc đã thiết kế từ `02_pipeline_architecture.md`; có cơ sở toán học vững chắc chống leakage đã kiểm chứng bằng thực nghiệm.

**Lưu ý kỹ thuật cho tương lai (chưa cần sửa ngay):** việc tách `fm_train`/`fm_valid` sau khi gộp hiện dựa vào so sánh `feature_matrix["Date"] <= train_w["Date"].max()` (`pipelines/run_train_baseline.py` dòng 93-94) — đúng cho 2 tập vì Date không trùng lặp giữa 2 tập liền kề, nhưng sẽ cần đổi sang cột nhãn tường minh (vd. `_split_label` gán trước khi gộp) khi Giai đoạn 5 mở rộng thêm `test_window` thật (3 khoảng Date liên tiếp cần phân biệt).

**Ảnh hưởng tới các module:** `src/sales_forecast/features/pipeline.py` (`build_feature_matrix` — không đổi code, chỉ chính thức hóa hành vi hiện có), `src/sales_forecast/features/lag_rolling.py` (không đổi, đã đúng), `pipelines/run_train_baseline.py` (ghi chú lưu ý kỹ thuật cho khi mở rộng Giai đoạn 5, không sửa ngay).

**Test case liên quan:** `tests/test_features/test_buffer_window_no_leakage.py` (mới) — xem `docs/05_test_plan.md` giả định A46.

---

## [2026-08-19] Ghi nhận `IsHoliday` lệch pha với đỉnh sales Christmas thật

**Bối cảnh:** Khảo sát trực tiếp `data/raw/features.csv` (đối chiếu `notebooks/00_eda.ipynb` mục 2b) cho thấy cột `IsHoliday` gắn cờ `True` cho đúng 13 tuần trong toàn bộ dữ liệu (2010-2013), tương ứng 4 dịp lễ Kaggle gốc (Super Bowl, Labor Day, Thanksgiving, Christmas). Riêng dịp Christmas, tuần được gắn cờ luôn là tuần **sau** ngày 25/12 (31/12, 30/12, 28/12) — tức tuần sales sụt mạnh hậu mua sắm — chứ không phải tuần chứa ngày 24/12 (đỉnh sales thật trước Christmas, không được gắn cờ). 3 dịp lễ còn lại (Super Bowl, Labor Day, Thanksgiving) không phát hiện lệch pha tương tự.

Danh sách đầy đủ 13 ngày (tuần kết thúc) được gắn `IsHoliday=True`:

| Ngày (tuần kết thúc) | Lễ tương ứng | Lệch pha? |
|---|---|---|
| 2010-02-12 | Super Bowl | Không |
| 2010-09-10 | Labor Day | Không |
| 2010-11-26 | Thanksgiving | Không |
| 2010-12-31 | Christmas | **Có** — đỉnh sales thật là tuần chứa 24/12, không được gắn cờ |
| 2011-02-11 | Super Bowl | Không |
| 2011-09-09 | Labor Day | Không |
| 2011-11-25 | Thanksgiving | Không |
| 2011-12-30 | Christmas | **Có** — tương tự 2010 |
| 2012-02-10 | Super Bowl | Không |
| 2012-09-07 | Labor Day | Không |
| 2012-11-23 | Thanksgiving | Không |
| 2012-12-28 | Christmas | **Có** — tương tự 2010 |
| 2013-02-08 | Super Bowl | Không |

**Các lựa chọn đã xem xét:**
- Giữ nguyên `IsHoliday` gốc làm feature duy nhất, không xử lý gì thêm — loại vì trọng số WMAE/WMAPE x5 (đã chốt 2026-08-18) sẽ đè lên đúng tuần sales thấp nhất tháng 12, còn tuần đỉnh sales thật (24/12) lại không được nhấn trọng số — sai lệch ý nghĩa nghiệp vụ của trọng số x5
- Sửa thẳng `IsHoliday` trong `data/raw/features.csv` — loại vì vi phạm invariant #7 (`data/raw/` bất biến, chỉ đọc)
- Ghi nhận lệch pha, để dành xử lý ở Giai đoạn 3 (Feature Engineering) bằng feature bổ sung tách biệt khỏi `IsHoliday` gốc

**Quyết định:** Ghi nhận chính thức hiện tượng lệch pha này là một giả định nghiệp vụ đã biết. CHƯA quyết định cơ chế xử lý cụ thể (vd. thêm feature `is_pre_christmas_peak` riêng, hay điều chỉnh trọng số WMAE/WMAPE x5 theo tuần chứa 24/12 thay vì dùng thẳng `IsHoliday`) — việc này để dành cho Giai đoạn 3 (Feature Engineering) khi thiết kế `configs/features.yaml`, tuân thủ invariant #2 (tách logic feature khỏi logic time-boundary) và invariant #9 (không fillna/gắn cờ mù quáng). Không sửa `IsHoliday` gốc trong dữ liệu hay trong trọng số WMAE/WMAPE x5 tại thời điểm ghi quyết định này.

**Lý do:** Đây là đặc điểm có sẵn của `IsHoliday` trong dữ liệu Kaggle gốc (không phải lỗi ingestion/pipeline của project), nhưng ảnh hưởng trực tiếp ý nghĩa của trọng số x5 đang dùng cho WMAE/WMAPE — cần ghi lại tường minh để không bị hiểu nhầm là bug khi review kết quả model, và để Giai đoạn 3 có căn cứ khi thiết kế feature block liên quan đến holiday.

**Ảnh hưởng tới các module:** `notebooks/00_eda.ipynb` (mục 2b, đã minh họa bằng biểu đồ có nhãn ngày), `src/sales_forecast/features/` (khi thiết kế feature block holiday ở Giai đoạn 3, cần cân nhắc hiện tượng này), `docs/01_ideation.md` mục 6 (trọng số WMAE/WMAPE x5 dựa trên `IsHoliday`). Không ảnh hưởng code hiện tại vì chưa có feature block holiday nào được implement.

**Test case liên quan:** Chưa cần — sẽ bổ sung vào `docs/05_test_plan.md` khi Giai đoạn 3 chốt cơ chế xử lý cụ thể cho feature holiday.

---

## Mẫu thêm quyết định mới

```markdown
## [YYYY-MM-DD] Tên quyết định

**Bối cảnh:** ...
**Các lựa chọn đã xem xét:** ...
**Quyết định:** ...
**Lý do:** ...
**Ảnh hưởng tới các module:** ...
**Test case liên quan (nếu có):** ...
```
