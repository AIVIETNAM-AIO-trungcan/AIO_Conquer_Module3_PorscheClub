# Cây thư mục Repo — Sales & Demand Forecasting

**AIO Conquer 2026 — Module 03 · Project 3.2**

> Nguyên tắc thiết kế: modular theo giai đoạn pipeline (khớp 1-1 với `02_pipeline_architecture.md`), tách biệt code khỏi notebook, mọi giả định đều có test tương ứng, config tách khỏi code cứng.
> rev2: bổ sung `app/` (dashboard Streamlit), `scripts/` (xác thực môi trường), `src/sales_forecast/evaluation/conformal.py` (khoảng tin cậy 95%), CatBoost trong `models/`.

```
AIO_Conquer_Module3_Project32/
│
├── CLAUDE.md                          # Hướng dẫn làm việc cho Claude/AI trong repo này
├── README.md                          # Giới thiệu ngắn, hướng dẫn chạy nhanh (quickstart)
├── pyproject.toml                     # Nguồn khai báo dependency DUY NHẤT (xem 06_environment_setup.md)
├── .gitignore                         # data/raw (data thô có thể lớn), data/interim, data/processed,
│                                       # .venv/, __pycache__, .env, docs/env_locks/*.txt (tùy chọn)
├── .env.example                       # Biến môi trường mẫu (đường dẫn data, seed, log level)
│
├── docs/                              # Tài liệu đặc tả — KHÔNG chứa code
│   ├── 00_decisions.md                # Log quyết định kiến trúc đã chốt (horizon, strategy...)
│   ├── 01_ideation.md
│   ├── 02_pipeline_architecture.md
│   ├── 03_data_io_diagram.md
│   ├── 04_repo_structure.md           # (chính tài liệu này)
│   ├── 05_test_plan.md
│   ├── 06_environment_setup.md        # Setup venv/pip, xác thực môi trường
│   ├── 07_dashboard_spec.md           # Đặc tả dashboard Streamlit
│   ├── 08_uncertainty_conformal.md    # Split Conformal Prediction 95%
│   ├── env_locks/                     # pip freeze trước mỗi lần nộp bài (xem 06_environment_setup.md §5)
│   └── specs/                         # Spec kỹ thuật ngắn cho từng block feature (1 file/block)
│       ├── spec_lag_rolling.md
│       ├── spec_calendar.md
│       ├── spec_markdown_promo.md
│       ├── spec_store_dept_encoding.md
│       └── spec_macro.md
│
├── configs/                           # Toàn bộ tham số cấu hình — KHÔNG hard-code trong src/
│   ├── data.yaml                      # đường dẫn file, split_date, horizon
│   ├── features.yaml                  # bật/tắt từng block feature, tham số lag/rolling window
│   ├── model_lightgbm.yaml
│   ├── model_xgboost.yaml
│   ├── model_catboost.yaml            # tham số CatBoost (hướng nâng cao)
│   ├── model_baseline.yaml
│   ├── optuna.yaml                    # search space, n_trials, timeout (cho LightGBM/XGBoost/CatBoost)
│   └── conformal.yaml                 # alpha=0.05, tỷ lệ tách calib_window từ valid_window
│
├── scripts/
│   └── check_env.py                   # Xác thực môi trường 1 lệnh (xem 06_environment_setup.md §4)
│
├── data/
│   ├── raw/                           # Bất biến — CHỈ ĐỌC, không sửa/ghi đè bằng tay
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── features.csv
│   │   └── stores.csv
│   ├── interim/                       # RawDataBundle đã validate + join (checkpoint trung gian)
│   ├── processed/                     # feature_matrix train/valid/test (parquet)
│   └── predictions/                   # xem 03_data_io_diagram.md §4-5 — versioned theo run_id
│       ├── runs/<run_id>/             # submission.csv, predictions_long.parquet của 1 lần chạy
│       └── latest_run.txt             # con trỏ run mới nhất có ghi vào predictions/
│
├── src/
│   └── sales_forecast/                # Package chính — import được, KHÔNG chạy trực tiếp notebook
│       ├── __init__.py
│       │
│       ├── ingestion/                 # Giai đoạn 1 — Data Ingestion & Validation
│       │   ├── __init__.py
│       │   ├── loaders.py             # đọc 4 CSV gốc
│       │   ├── schema.py              # định nghĩa schema kỳ vọng (pandera)
│       │   └── validators.py          # Data Contract: raise lỗi nếu sai schema/toàn vẹn thời gian
│       │
│       ├── splitting/                 # Giai đoạn 2 — Temporal Split
│       │   ├── __init__.py
│       │   └── temporal_split.py      # cắt train/valid/test theo as_of_date, KHÔNG random
│       │
│       ├── features/                  # Giai đoạn 3 — Feature Engineering (block độc lập)
│       │   ├── __init__.py
│       │   ├── base.py                # interface chung: FeatureBlock.transform(df, as_of_date)
│       │   ├── lag_rolling.py
│       │   ├── calendar.py
│       │   ├── markdown_promo.py
│       │   ├── store_dept_encoding.py
│       │   ├── macro.py
│       │   └── pipeline.py            # ghép các block theo config features.yaml
│       │
│       ├── models/                    # Giai đoạn 4-6 — Baseline, Training, Tuning
│       │   ├── __init__.py
│       │   ├── baseline.py            # Naive same-week-last-year, Decision Tree đơn giản
│       │   ├── tree_models.py         # DT, RF, AdaBoost/GBM wrapper thống nhất interface
│       │   ├── boosting_models.py     # LightGBM (chính), XGBoost (so sánh)
│       │   ├── catboost_model.py      # CatBoost (so sánh, hướng nâng cao — xem 01_ideation.md §8)
│       │   ├── registry.py            # đăng ký model theo tên, dùng chung interface .fit/.predict
│       │   └── tuning.py              # Optuna, tái dùng CV splitter của evaluation/
│       │
│       ├── evaluation/                # Giai đoạn 7 — Evaluation Layer DÙNG CHUNG mọi model
│       │   ├── __init__.py
│       │   ├── cv_splitter.py         # TimeSeriesSplit / walk-forward, dùng chung cho train & Optuna
│       │   ├── metrics.py             # WMAE, WMAPE (trọng số IsHoliday), MAE, RMSE
│       │   ├── conformal.py           # Giai đoạn 6b — Split Conformal Prediction (xem 08_uncertainty_conformal.md)
│       │   └── error_analysis.py      # breakdown theo horizon/segment/cold-start/coverage
│       │
│       ├── explainability/            # Giai đoạn 8 — TreeSHAP
│       │   ├── __init__.py
│       │   ├── shap_runner.py
│       │   └── stability.py           # so sánh SHAP rank giữa các fold
│       │
│       ├── reporting/                 # Giai đoạn 9 — Reporting & Packaging
│       │   ├── __init__.py
│       │   └── report_builder.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── io.py                  # ghi/đọc atomic (JSON/parquet/CSV), không biết gì về "run"
│           ├── run_tracking.py        # RunContext, run_id, latest_run.txt, run_history.csv — versioning output
│           ├── logging_config.py
│           └── seed.py                # cố định random seed toàn pipeline
│
├── pipelines/                         # Script orchestration — điểm vào chạy end-to-end
│   ├── run_ingestion.py
│   ├── run_feature_engineering.py
│   ├── run_train_baseline.py
│   ├── run_train_full.py              # DT → RF → AdaBoost/GBM → LightGBM → XGBoost → CatBoost
│   ├── run_optuna_tuning.py
│   ├── run_conformal_calibration.py   # Giai đoạn 6b
│   ├── run_evaluation.py
│   ├── run_shap.py
│   └── run_end_to_end.py              # gọi tuần tự toàn bộ, tương ứng sơ đồ 02_pipeline_architecture.md
│
├── app/                                # Dashboard Streamlit — LỚP TRÌNH BÀY, không train model
│   ├── dashboard.py                    # Entry point: streamlit run app/dashboard.py
│   ├── theme.py                        # Bảng màu 1 model = 1 màu, dùng chung mọi tab
│   ├── data_loader.py                  # Đọc reports/, data/predictions/ — @st.cache_data
│   └── components/
│       ├── model_comparison.py         # Tab 1
│       ├── series_forecast.py          # Tab 2
│       ├── error_analysis.py           # Tab 3
│       ├── shap_explainer.py           # Tab 4
│       └── conformal_coverage.py       # Tab 5
│
├── notebooks/                         # CHỈ dùng để khám phá / trực quan hóa, KHÔNG chứa logic sản xuất
│   ├── 00_eda.ipynb
│   ├── 01_feature_exploration.ipynb
│   └── 02_shap_exploration.ipynb
│
├── tests/                             # Đối xứng 1-1 với src/ và app/ — xem chi tiết 05_test_plan.md
│   ├── conftest.py                    # fixture: sample data giả lập nhỏ (không đụng data/raw thật)
│   ├── test_ingestion/
│   │   ├── test_schema_validation.py
│   │   └── test_join_integrity.py
│   ├── test_splitting/
│   │   └── test_temporal_split_no_leakage.py
│   ├── test_features/
│   │   ├── test_lag_rolling_no_future_leak.py
│   │   ├── test_markdown_flag.py
│   │   ├── test_cold_start_handling.py
│   │   └── test_feature_block_independence.py
│   ├── test_evaluation/
│   │   ├── test_cv_splitter_order.py
│   │   ├── test_metrics_correctness.py
│   │   └── test_conformal_prediction.py   # xem 08_uncertainty_conformal.md
│   ├── test_models/
│   │   └── test_model_interface_consistency.py
│   ├── test_explainability/
│   │   └── test_shap_stability.py
│   ├── test_utils/
│   │   └── test_run_tracking.py           # xem 05_test_plan.md A25-A33
│   └── test_app/
│       ├── test_data_loader.py            # xem 07_dashboard_spec.md
│       └── test_theme.py
│
├── reports/                           # Output cuối: metrics, hình ảnh SHAP, optuna trials, coverage
│   ├── runs/<run_id>/                 # 1 thư mục/lần chạy — figures/, metrics/, manifest.json
│   ├── run_history.csv                # 1 dòng/model/run, append-only — lịch sử WMAE/WMAPE mọi lần chạy
│   └── latest_run.txt                 # con trỏ run mới nhất có ghi vào reports/
│
└── Presentation/                      # Giữ nguyên theo cấu trúc hiện có của team
    └── W1_t2/
        └── AIO_Module3_Sales_Forecasting_Ideation_Pipeline.pptx
```

## Nguyên tắc đi kèm cây thư mục

1. **`src/sales_forecast/` là package thật** (có `pyproject.toml`, cài bằng `pip install -e .`) — không import chéo lung tung qua đường dẫn tương đối trong notebook.
2. **`pipelines/` chỉ orchestration, không chứa logic nghiệp vụ** — mọi logic thật nằm trong `src/`, giúp test được (`tests/`) mà không cần chạy toàn bộ pipeline.
3. **`notebooks/` không bao giờ là nguồn sự thật (source of truth)** — mọi phát hiện hay từ notebook cần được "tốt nghiệp" thành module trong `src/` + test tương ứng trước khi coi là kết quả chính thức.
4. **`configs/` tách khỏi code** — thay đổi horizon, bật/tắt feature block, search space Optuna, hay tham số conformal (`alpha`) không cần sửa code Python.
5. **`data/raw/` bất biến** — pipeline chỉ đọc, không bao giờ ghi đè; mọi output trung gian đi vào `data/interim/` hoặc `data/processed/` (đã có trong `.gitignore`, không commit data lớn lên Git).
6. **`tests/` đối xứng cấu trúc với `src/` và `app/`** — mỗi module nghiệp vụ quan trọng có ít nhất 1 file test tương ứng, giảm rủi ro tech debt khi refactor.
7. **`app/` không import trực tiếp `lightgbm.train`/`.fit()` của bất kỳ model nào** — chỉ đọc file kết quả đã có trong `reports/`/`data/predictions/`. Việc này giữ dashboard nhẹ, mở nhanh, và không làm lệch kết quả đã báo cáo mỗi lần mentor mở app (xem `07_dashboard_spec.md` §1).
8. **`scripts/check_env.py`** là điểm kiểm tra môi trường duy nhất — không có script xác thực môi trường thứ hai rải rác nơi khác trong repo.
9. **Output pipeline versioned theo `run_id`, không ghi đè.** Mỗi lần chạy pipeline (`pipelines/run_*.py`) sinh 1 `run_id` qua `src/sales_forecast/utils/run_tracking.py`, ghi vào `reports/runs/<run_id>/` và `data/predictions/runs/<run_id>/` thay vì path cố định. `latest_run.txt` (2 pointer độc lập cho `reports/` và `data/predictions/`) luôn trỏ đúng run thành công gần nhất để dashboard đọc mặc định mà không cần chọn `run_id` thủ công; `run_history.csv` giữ lịch sử WMAE/WMAPE mọi lần chạy để so sánh cải thiện qua thời gian. Chi tiết: `03_data_io_diagram.md` §5, `docs/00_decisions.md`.
