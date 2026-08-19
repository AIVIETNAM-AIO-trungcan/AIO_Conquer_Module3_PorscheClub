# Sơ đồ giải thuật — Kiến trúc Pipeline 10 Giai đoạn

**AIO Conquer 2026 — Module 03 · Project 3.2**

> Mỗi giai đoạn = 1 module độc lập, input/output rõ ràng, không rò rỉ thông tin giữa các giai đoạn.
> Tham chiếu quyết định chưa chốt (horizon/strategy) ở `01_ideation.md` mục 4 — sơ đồ dưới đây **không phụ thuộc** vào lựa chọn A/B/C, chỉ giai đoạn 3 và 5 có nhánh rẽ nội bộ.

---

## 1. Sơ đồ tổng thể (flowchart)

```mermaid
flowchart TD
    subgraph G0["0-3 · DỮ LIỆU — Chống Leakage"]
        S0["0. Problem Framing<br/>Chốt horizon, chiến lược, metric (WMAPE)"]
        S1["1. Data Ingestion & Validation<br/>Schema contract<br/>kiểm tra toàn vẹn thời gian"]
        S2["2. Temporal Split<br/>Cắt mốc train/valid/test<br/>TRƯỚC feature engineering"]
        S3["3. Feature Engineering<br/>Lag · Rolling · Calendar<br/>MarkDown/Promo · Store-Dept encoding"]
        S0 --> S1 --> S2 --> S3
    end

    subgraph G1["4-6 · MÔ HÌNH — Tree progression"]
        S4["4. Baseline Models<br/>Naive (cùng kỳ tuần trước) + Decision Tree"]
        S5["5. Model Training<br/>DT → RF → AdaBoost/GBM<br/>→ LightGBM (chính) → XGBoost (so sánh)"]
        S6["6. Optuna Tuning<br/>Walk-forward validation, log toàn bộ trial"]
        S6b["6b. Conformal Calibration<br/>Split Conformal 95%<br/>calib_window riêng"]
        S4 --> S5 --> S6 --> S6b
    end

    subgraph G2["7-9 · KIỂM ĐỊNH — Giải thích & Bàn giao"]
        S7["7. Evaluation & Error Analysis<br/>Theo horizon · fast/slow Dept<br/>có/không MarkDown · cold-start · coverage 95%"]
        S8["8. TreeSHAP<br/>Feature importance<br/>+ dependence plot + ổn định qua fold"]
        S9["9. Reporting & Packaging<br/>Báo cáo kết quả · Dashboard Streamlit<br/>đóng gói module tái sử dụng"]
        S7 --> S8 --> S9
    end

    S3 --> S4
    S6b --> S7

    style G0 fill:#fdece0,stroke:#d98a55
    style G1 fill:#e3f0e6,stroke:#5a9e6f
    style G2 fill:#e6ecf5,stroke:#5578ad
    style S6b fill:#fff3cd,stroke:#c9a227
```

---

## 2. Chi tiết Giai đoạn 0–3 — Nền tảng chống Data Leakage

```mermaid
flowchart LR
    A["train.csv<br/>test.csv<br/>features.csv<br/>stores.csv"] --> B["1. Data Ingestion<br/>& Validation"]
    B -->|"schema hợp lệ?<br/>date liên tục?<br/>(Store,Dept) hợp lệ?"| C{"Data Contract<br/>pass?"}
    C -->|Fail| B_err["Raise DataContractError<br/>(dừng pipeline, không âm thầm bỏ qua)"]
    C -->|Pass| AGG["1b. Aggregate<br/>(Store,Dept,Date) -> (Store,Date)<br/>SUM Weekly_Sales theo Dept"]
    AGG --> D["2. Temporal Split<br/>as_of_date cố định"]
    D --> D1["train_window<br/>(≤ split_date - horizon)"]
    D --> D2["valid_window<br/>(split_date - horizon, split_date]"]
    D --> D3["test_window<br/>(test.csv thật, không có target)"]
    D1 --> E["3. Feature Engineering<br/>(hàm thuần, nhận as_of_date)"]
    D2 --> E
    D3 --> E
    E --> E1["Block: Lag/Rolling<br/>(chỉ dùng dữ liệu ≤ t-1)"]
    E --> E2["Block: Calendar<br/>(IsHoliday, week-of-year)"]
    E --> E3["Block: MarkDown/Promo<br/>(join features.csv + flag has_markdown)"]
    E --> E4["Block: Store-Dept encoding<br/>(categorical native LightGBM)"]
    E --> E5["Block: Macro<br/>(CPI, Unemployment, Temperature, Fuel_Price)"]
    E1 & E2 & E3 & E4 & E5 --> F["feature_matrix<br/>(train/valid/test)"]

    style B_err fill:#f8d7da,stroke:#c0392b
```

**Quy tắc bất biến (invariant) của Giai đoạn 0–3:**
- **(Cập nhật 2026-08-19)** Giai đoạn 1b (Aggregate) đổi đơn vị dự báo từ (Store, Dept, Date) sang (Store, Date) NGAY SAU Data Contract, TRƯỚC Temporal Split — xem `docs/00_decisions.md` "Đổi đơn vị dự báo". Đây là thay đổi đơn vị quan sát, không phải feature, nên phải xảy ra sớm nhất.
- Cắt mốc thời gian **trước**, tính feature **sau** — không đảo thứ tự.
- Mỗi block feature là **hàm thuần** `f(df, as_of_date) -> df_features`, không có side-effect, không nhìn quá khứ toàn cục nếu chưa qua as_of_date.
- Data Contract (giai đoạn 1) là nơi **duy nhất** kiểm tra schema — các giai đoạn sau không lặp lại kiểm tra rải rác.

---

## 3. Chi tiết Giai đoạn 4–6 — Modeling & Tuning

```mermaid
flowchart TD
    F["feature_matrix"] --> G4["4. Baseline<br/>Naive same-week-last-year<br/>+ Decision Tree đơn giản"]
    F --> G5a["5a. Decision Tree"]
    F --> G5b["5b. Random Forest"]
    F --> G5c["5c. AdaBoost / Gradient Boost"]
    F --> G5d["5d. LightGBM (chính)"]
    F --> G5e["5e. XGBoost (so sánh)"]
    G4 & G5a & G5b & G5c & G5d & G5e --> EVAL["Evaluation Layer<br/>(dùng chung 1 cơ chế TimeSeriesSplit)"]
    EVAL --> G6["6. Optuna Tuning<br/>(trên LightGBM/XGBoost,<br/>tái dùng walk-forward của Evaluation)"]
    G6 --> G6a["So sánh trước/sau tối ưu<br/>(bắt buộc báo cáo cả hai)"]
    G6a --> G6b["6b. Conformal Calibration<br/>tách calib_window từ valid_window<br/>KHÔNG dùng test_window"]
    G6b --> G6c["Sinh y_pred_lower / y_pred_upper<br/>cho mọi model<br/>mọi dòng dự đoán"]

    style EVAL fill:#e3f0e6,stroke:#5a9e6f
    style G6b fill:#fff3cd,stroke:#c9a227
```

**Quy tắc bất biến:**
- Mọi model (kể cả baseline) đi qua **cùng một Evaluation Layer** — không viết logic đánh giá riêng cho từng model.
- Optuna chỉ được thấy `train_window` + `valid_window`, không bao giờ chạm `test_window`.
- Optuna tái dùng đúng cơ chế walk-forward split của Evaluation Layer (giai đoạn 6 gọi lại module của giai đoạn 4-6, không tự viết CV riêng).
- **Conformal Calibration (6b) chạy SAU khi Optuna đã chốt best trial**, dùng phần `calib_window` tách riêng từ `valid_window` (không phải phần đã dùng cho Optuna) — chi tiết đầy đủ và lý do thiết kế ở `08_uncertainty_conformal.md`.

---

## 4. Chi tiết Giai đoạn 7–9 — Kiểm định, Giải thích, Bàn giao

```mermaid
flowchart TD
    M["Model đã tối ưu<br/>+ hiệu chỉnh conformal<br/>(best trial Optuna + calibrator)"] --> S7["7. Evaluation & Error Analysis"]
    S7 --> S7a["Theo forecast horizon (h=1..H)"]
    S7 --> S7b["Theo fast/slow-moving Dept"]
    S7 --> S7c["Theo có/không MarkDown"]
    S7 --> S7d["Theo cold-start vs. có lịch sử"]
    S7 --> S7e["Trước/sau Optuna (bắt buộc so sánh)"]
    S7 --> S7f["7f. Coverage Check<br/>Empirical coverage vs. 95% mục tiêu"]

    M --> S8["8. TreeSHAP"]
    S8 --> S8a["Global feature importance<br/>(so sánh LightGBM/XGBoost)"]
    S8 --> S8b["Dependence plot<br/>(MarkDown → SHAP value có hợp lý không?)"]
    S8 --> S8c["Ổn định SHAP qua các fold<br/>(so sánh rank importance)"]

    S7a & S7b & S7c & S7d & S7e & S7f & S8a & S8b & S8c --> S9["9. Reporting & Packaging"]
    S9 --> S9a["Báo cáo kỹ thuật (docs/)"]
    S9 --> S9b["Đóng gói module tái sử dụng<br/>(src/ dạng package<br/>không notebook rời rạc)"]
    S9 --> S9c["Dashboard Streamlit<br/>(app/dashboard.py — đọc reports/<br/>KHÔNG train lại)"]
```

---

## 5. Bảng tóm tắt Input → Output theo từng giai đoạn

| Giai đoạn | Input | Output | Rủi ro chính cần test |
|---|---|---|---|
| 0. Problem Framing | Tài liệu đề bài, checklist | `docs/00_decisions.md` | Quyết định không được ghi lại → tranh cãi giữa chừng |
| 1. Data Ingestion | 4 CSV thô | `RawDataBundle` đã validate | Schema sai, ngày trùng/thiếu, (Store,Dept) không hợp lệ |
| 1b. Aggregate *(2026-08-19)* | `RawDataBundle` đã validate | Train/test đã gộp về (Store, Date), Dept không còn tồn tại | Aggregate SUM sai (double-count khi join lại), IsHoliday không nhất quán bị `.first()` chọn sai |
| 2. Temporal Split | `RawDataBundle` + `as_of_date`/`split_date` | `train_window`, `valid_window`, `test_window` | Rò rỉ dữ liệu tương lai vào train |
| 3. Feature Engineering | 3 window ở trên | `feature_matrix` (mỗi block riêng) | Lag/rolling dùng dữ liệu > t−1; join `features.csv` nhân bản dòng |
| 4. Baseline | `feature_matrix` (train) | `baseline_predictions` | Baseline bị bỏ qua, không có mốc so sánh |
| 5. Model Training | `feature_matrix` | Model đã fit (DT/RF/AdaBoost-GBM/LightGBM/XGBoost) | Train/predict không dùng cùng bộ feature |
| 6. Optuna Tuning | `train_window`, `valid_window`, search space | `best_params`, `trial_history` | Tuning nhìn thấy test_window (leakage gián tiếp) |
| 6b. Conformal Calibration | Model đã tối ưu + `calib_window` (tách từ `valid_window`) | Calibrator (giá trị `q`), sinh `y_pred_lower/upper` | Dùng nhầm `train_window` hoặc `test_window` để hiệu chỉnh; dùng `np.quantile` trần thay vì hiệu chỉnh hữu hạn mẫu |
| 7. Evaluation | Model + `test_window`/`valid_window` + calibrator | Bảng metric theo nhiều lát cắt + coverage report | Chỉ báo cáo 1 con số tổng, che giấu yếu điểm theo nhóm; không kiểm tra coverage thực tế |
| 8. TreeSHAP | Model đã tối ưu + `feature_matrix` | SHAP values, dependence plot (theo từng model) | Diễn giải nhân quả sai (confounding giá & khuyến mãi) |
| 9. Reporting | Toàn bộ kết quả trên | Báo cáo + package `src/` + Dashboard Streamlit | Code chỉ tồn tại trong notebook, không tái dùng được; dashboard tự train lại thay vì chỉ đọc kết quả |

Chi tiết schema từng bảng dữ liệu và luồng input/output đầy đủ: xem `03_data_io_diagram.md`. Chi tiết Conformal Calibration: `08_uncertainty_conformal.md`. Chi tiết Dashboard: `07_dashboard_spec.md`. Chi tiết môi trường: `06_environment_setup.md`.
