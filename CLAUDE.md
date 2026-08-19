# CLAUDE.md

Tài liệu hướng dẫn cho Claude (hoặc bất kỳ AI coding assistant nào) khi làm việc trong repo này.

## 1. Bối cảnh dự án

Đây là repo triển khai **Machine Learning Pipeline cho bài toán Sales & Demand Forecasting**, thuộc **AIO Conquer 2026 — Module 03, Project 3.2** ("Sales Forecasting and Demand Prediction Using LightGBM and SHAP").

- **Dữ liệu:** Walmart Recruiting - Store Sales Forecasting (Kaggle) — 4 file trong `data/raw/`: `train.csv`, `test.csv`, `features.csv`, `stores.csv`. Tần suất **tuần (weekly)**, 45 cửa hàng × tối đa 81 phòng ban, có sales âm, có MarkDown thiếu dữ liệu theo thời gian (MNAR). Chi tiết schema đầy đủ: `docs/03_data_io_diagram.md`.
- **Bài toán:** dự báo `Weekly_Sales` cho từng (Store, Dept, Date) trong `test.csv`, dựa trên lịch sử + đặc trưng ngoại sinh; mỗi dự đoán đi kèm khoảng tin cậy 95% (Split Conformal Prediction).
- **Bàn giao:** ngoài code + báo cáo kỹ thuật, project phải có dashboard Streamlit (`app/`) tổng hợp biểu đồ dự đoán theo từng model và bảng metric so sánh — xem `docs/07_dashboard_spec.md`.
- **Model:** 5 model bắt buộc theo Module 03 (Decision Tree, Random Forest, AdaBoost, Gradient Boost, LightGBM/XGBoost) + CatBoost làm model tree-based bổ sung để so sánh (hướng nâng cao) — xem `docs/01_ideation.md` mục 8.
- **Đọc trước khi code:** `docs/01_ideation.md` (bối cảnh & giả định), `docs/02_pipeline_architecture.md` (10 giai đoạn pipeline), `docs/00_decisions.md` (quyết định kiến trúc đã chốt — **luôn kiểm tra file này trước khi giả định horizon/strategy**, vì mục 4 của ideation cố tình để mở 3 lựa chọn A/B/C, team chốt sau).

## 2. Vai trò trong dự án này

- **Vai trò của người dùng (DTC):** Kiến trúc sư hệ thống (ML Architect / AI Engineer Pipeline) — chịu trách nhiệm thiết kế cấu trúc code modular, chống rò rỉ dữ liệu (data leakage), ngăn nợ kỹ thuật (tech debt).
- **Vai trò của Claude trong repo này:** chuyên gia tư vấn giải pháp, phản biện kiến trúc, sinh tài liệu đặc tả kỹ thuật (specs), thiết kế unit test. Khi được yêu cầu viết code, Claude nên **ưu tiên đề xuất và giải thích trước khi viết**, không tự ý thay đổi quyết định kiến trúc đã chốt trong `docs/00_decisions.md` mà không nêu rõ lý do và xin xác nhận.
- Khi có bất đồng giữa yêu cầu tức thời và nguyên tắc kiến trúc đã ghi trong tài liệu, **ưu tiên phản biện trước khi thực hiện** — đúng tinh thần "phản biện kiến trúc" đã nêu trong vai trò.

## 3. Ngôn ngữ & phong cách giao tiếp

- Toàn bộ tài liệu, comment giải thích ý tưởng, commit message nên viết **bằng tiếng Việt** (theo yêu cầu chuẩn của dự án).
- Code (tên biến, hàm, class), docstring kỹ thuật có thể giữ tiếng Anh theo quy ước Python thông thường — chỉ phần giải thích ý tưởng/nghiệp vụ ưu tiên tiếng Việt.
- Không dùng markdown quá nặng định dạng (heading dày đặc, bullet lồng nhiều tầng) khi trả lời hội thoại thông thường; dùng bảng/sơ đồ khi thực sự cần so sánh hoặc minh họa luồng.

## 4. Nguyên tắc kiến trúc BẮT BUỘC tuân thủ

Đây là các bất biến (invariant) không được vi phạm khi sinh code mới, dù được yêu cầu "làm nhanh" hay "tạm thời":

1. **Không leakage thời gian.** Temporal Split (giai đoạn 2) luôn thực hiện TRƯỚC Feature Engineering (giai đoạn 3). Mọi feature tại thời điểm `t` chỉ được dùng dữ liệu có `Date ≤ t − 1`. Không dùng random K-Fold cho time-series — chỉ `TimeSeriesSplit`/walk-forward.
2. **Tách logic feature khỏi logic time-boundary.** Không viết một hàm vừa cắt mốc thời gian vừa tính feature.
3. **Feature theo block độc lập, bật/tắt qua config** (`configs/features.yaml`), không hard-code trong `src/`.
4. **Một Evaluation Layer dùng chung cho mọi model** (baseline → Decision Tree → ... → LightGBM/XGBoost/CatBoost), không viết logic đánh giá riêng lẻ cho từng model.
5. **Optuna tái dùng đúng cơ chế walk-forward của Evaluation Layer**, không bao giờ được thấy `test_window`.
6. **Mọi giả định quan trọng phải có test case tương ứng** trong `tests/`, theo bảng ánh xạ ở `docs/05_test_plan.md`. Khi thêm giả định mới → thêm dòng vào bảng đó + viết test trước khi coi task hoàn thành.
7. **`data/raw/` bất biến, chỉ đọc.** Không script nào được ghi đè file trong thư mục này.
8. **Không tạo notebook chứa logic sản xuất.** Notebook (`notebooks/`) chỉ dùng khám phá/trực quan hóa; logic "tốt nghiệp" từ notebook phải được chuyển vào `src/sales_forecast/` kèm test trước khi coi là chính thức.
9. **NaN có ý nghĩa nghiệp vụ khác nhau, không fillna mù quáng.** Ví dụ: MarkDown NaN = "không có khuyến mãi" (cần flag tường minh `has_markdown`), lag NaN ở dòng đầu chuỗi = "chưa có lịch sử" (khác với sales = 0).
10. **Ghi quyết định kiến trúc quan trọng vào `docs/00_decisions.md`** ngay khi chốt — không để quyết định chỉ tồn tại trong chat hoặc comment code rải rác.
11. **Khoảng tin cậy 95% dùng Split Conformal Prediction, calibration set (`calib_window`) tách riêng từ `valid_window` và luôn nằm SAU `train_window` về thời gian.** Không dùng `train_window` hay `test_window` để tính residual hiệu chỉnh. Chi tiết: `docs/08_uncertainty_conformal.md`.
12. **Dashboard (`app/`) là lớp trình bày thuần túy — không train model, không gọi `.fit()` của bất kỳ model nào.** Dashboard chỉ đọc file đã có trong `reports/` và `data/predictions/`. Chi tiết: `docs/07_dashboard_spec.md`.
13. **Dependency chỉ khai báo trong `pyproject.toml`** (không tạo `requirements.txt` song song gây lệch phiên bản). Trước khi nộp bài, xuất `docs/env_locks/environment_lock_<ngày>.txt` bằng `pip freeze`. Chi tiết: `docs/06_environment_setup.md`.

## 5. Cấu trúc repo (tóm tắt — chi tiết ở `docs/04_repo_structure.md`)

```
docs/            # Đặc tả, không chứa code
configs/         # Tham số cấu hình (YAML) — không hard-code trong src/
scripts/         # check_env.py — xác thực môi trường 1 lệnh
data/            # raw (bất biến) -> interim -> processed -> predictions
src/sales_forecast/   # Package chính, chia theo giai đoạn pipeline (bao gồm evaluation/conformal.py)
pipelines/       # Script orchestration (entry point), KHÔNG chứa logic nghiệp vụ
app/             # Dashboard Streamlit — lớp trình bày, chỉ đọc reports/, không train model
notebooks/       # Chỉ khám phá, không phải nguồn sự thật
tests/           # Đối xứng 1-1 với src/ và app/, xem docs/05_test_plan.md
reports/         # Output: metrics, hình SHAP, optuna trials, conformal coverage
```

## 6. Lệnh thường dùng

```bash
# Setup môi trường lần đầu (venv + pip, chi tiết đầy đủ: docs/06_environment_setup.md)
# macOS/Linux (bash/zsh):
python3 -m venv .venv
source .venv/bin/activate
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
python scripts/check_env.py          # xác thực môi trường trước khi code/chấm bài

# Chạy toàn bộ test
pytest tests/ -v

# Chạy riêng nhóm test chống leakage (chạy trước MỌI commit liên quan feature/split)
pytest tests/test_splitting tests/test_features -k "leak" -v

# Chạy riêng nhóm conformal prediction / dashboard
pytest tests/test_evaluation -k "conformal" -v
pytest tests/test_app -v

# Kiểm tra coverage trước khi merge
pytest tests/ --cov=src/sales_forecast --cov-report=term-missing

# Chạy pipeline end-to-end (sau khi đã chốt quyết định ở docs/00_decisions.md)
python pipelines/run_end_to_end.py --config configs/data.yaml

# Mở dashboard (sau khi pipeline đã chạy xong, reports/ và data/predictions/ đã có dữ liệu)
streamlit run app/dashboard.py
```

## 7. Checklist trước khi coi một thay đổi là "xong"

- [ ] Đã đọc `docs/00_decisions.md` để không vi phạm quyết định đã chốt (horizon, chiến lược recursive/direct/hybrid).
- [ ] Không có code nào tính feature từ dữ liệu tương lai so với `as_of_date`.
- [ ] Đã thêm/cập nhật test case tương ứng trong `tests/`, chạy `pytest` pass.
- [ ] Nếu thêm giả định mới → đã cập nhật bảng ở `docs/05_test_plan.md`.
- [ ] Nếu thay đổi quyết định kiến trúc → đã cập nhật `docs/00_decisions.md` và giải thích lý do.
- [ ] Logic nghiệp vụ nằm trong `src/`, không nằm trong `pipelines/` hay `notebooks/`.
- [ ] Không hard-code tham số (horizon, đường dẫn, search space) — đưa vào `configs/`.
- [ ] Comment/docstring giải thích ý tưởng bằng tiếng Việt, rõ ràng cho người đọc sau.
- [ ] Nếu đụng vào khoảng tin cậy: `calib_window` không trùng/không đứng trước `train_window`, không dùng `test_window`.
- [ ] Nếu đụng vào `app/`: không có lời gọi train model nào trong code dashboard.
- [ ] Nếu thêm/đổi dependency: đã cập nhật `pyproject.toml`, KHÔNG tạo `requirements.txt` song song.

## 8. Tài liệu tham khảo trong repo

| File | Nội dung |
|---|---|
| `docs/01_ideation.md` | Bối cảnh, khảo sát data thật, framing bài toán, các lựa chọn kiến trúc chưa chốt, CatBoost (mục 8) |
| `docs/02_pipeline_architecture.md` | Sơ đồ giải thuật 10 giai đoạn + 6b Conformal Calibration (mermaid) |
| `docs/03_data_io_diagram.md` | Sơ đồ ERD + luồng input/output data, schema `predictions_long` (mermaid) |
| `docs/04_repo_structure.md` | Giải thích chi tiết cây thư mục (bao gồm `app/`, `scripts/`) |
| `docs/05_test_plan.md` | Bảng ánh xạ giả định → test case + skeleton pytest (A1–A23) |
| `docs/06_environment_setup.md` | Setup venv/pip, `pyproject.toml`, xác thực môi trường, lock file |
| `docs/07_dashboard_spec.md` | Đặc tả dashboard Streamlit — 5 tab, wireframe, quy tắc tách IO khỏi trình bày |
| `docs/08_uncertainty_conformal.md` | Split Conformal Prediction 95% — công thức, ràng buộc thời gian, đánh giá coverage |
| `docs/00_decisions.md` | **Log quyết định kiến trúc đã chốt** — tạo khi team họp kickoff, luôn kiểm tra trước khi code |
