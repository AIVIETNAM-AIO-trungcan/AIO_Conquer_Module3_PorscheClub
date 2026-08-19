# Setup Môi trường — venv + pip

**AIO Conquer 2026 — Module 03 · Project 3.2**

> Mục tiêu: mọi thành viên team và mentor chạy được **chính xác cùng một môi trường** trên Windows/macOS/Linux, không cần cài Anaconda/Docker. Dùng `venv` (built-in Python) + `pyproject.toml` làm nguồn khai báo dependency duy nhất — không có `requirements.txt` trùng lặp gây lệch phiên bản.

---

## 1. Yêu cầu tiên quyết

| Thành phần | Phiên bản yêu cầu | Kiểm tra |
|---|---|---|
| Python | 3.10 – 3.12 (khuyến nghị 3.11) | `python --version` |
| pip | ≥ 23.0 | `pip --version` |
| Git | bất kỳ bản gần đây | `git --version` |

**Vì sao giới hạn Python 3.10–3.12:** LightGBM/XGBoost đều publish wheel sẵn (prebuilt) cho dải version này trên cả 3 hệ điều hành, tránh trường hợp mentor phải tự build từ source (nguồn lỗi cài đặt phổ biến nhất trong các project ML nhóm).

---

## 2. Các gói phụ thuộc chính (`pyproject.toml`)

File `pyproject.toml` ở gốc repo là **nguồn khai báo dependency duy nhất**. Dùng khoảng version (không pin cứng 1 con số) để vẫn nhận bản vá lỗi nhưng tránh breaking change ở major version:

| Nhóm | Package | Khoảng version | Lý do |
|---|---|---|---|
| Core data | `pandas`, `numpy`, `pyarrow` | pandas ≥2.1,<3.0 · numpy ≥1.26,<2.0 | numpy <2.0 vì một số bản LightGBM/SHAP cũ chưa tương thích đầy đủ numpy 2.x tại thời điểm viết tài liệu — kiểm tra lại khi nâng cấp |
| Model cây | `scikit-learn`, `lightgbm`, `xgboost` | xem bảng dưới | 4 model chính theo scope Module 03 |
| Tối ưu & giải thích | `optuna`, `shap` | optuna ≥3.6,<4.0 · shap ≥0.45,<0.46 | pin chặt hơn cho `shap` vì API TreeExplainer đôi khi đổi giữa minor version |
| Validate & IO | `pandera`, `pyyaml` | — | Data Contract (giai đoạn 1) dùng `pandera` để validate schema bằng code, không kiểm tra thủ công |
| Dashboard | `streamlit`, `plotly` | ≥1.35 / ≥5.20 | xem `07_dashboard_spec.md` |
| Dev/test | `pytest`, `pytest-cov`, `ruff` | trong nhóm `[dev]` | không cài ở môi trường "chỉ chạy dashboard" của mentor nếu muốn nhẹ hơn |

**Bằng chứng version đã kiểm tra (18/8/2026):** LightGBM bản ổn định mới nhất là 4.6.0 (phát hành 14/2/2025). Khoảng version trong `pyproject.toml` được chọn dựa trên mốc này — xem nguồn ở cuối tài liệu.

---

## 3. Hướng dẫn cài đặt — 3 hệ điều hành

### 3.1. Windows (PowerShell)

```powershell
git clone <repo-url> sales-forecast
cd sales-forecast

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"

# Kiểm tra cài đặt thành công
python -c "import lightgbm, xgboost, optuna, shap, streamlit; print('OK - moi truong san sang')"
```

### 3.2. macOS / Linux (bash/zsh)

```bash
git clone <repo-url> sales-forecast
cd sales-forecast

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

python -c "import lightgbm, xgboost, optuna, shap, streamlit; print('OK - moi truong san sang')"
```

### 3.3. Môi trường "chỉ đánh giá" cho mentor (không cần dev tools)

```bash
python -m venv .venv
source .venv/bin/activate   # hoặc .\.venv\Scripts\Activate.ps1 trên Windows
pip install -e .            # KHÔNG có [dev] -> nhẹ hơn, đủ để chạy pipeline + dashboard
```

---

## 4. Xác thực môi trường (dùng chung cho mọi thành viên)

Script `scripts/check_env.py` (xem mục 6) chạy tự động các bước sau, mentor chỉ cần gõ 1 lệnh:

```bash
python scripts/check_env.py
```

Script kiểm tra:
1. Version Python nằm trong khoảng cho phép (3.10–3.12).
2. Toàn bộ package trong `pyproject.toml` import được, in ra version thực tế đã cài.
3. `data/raw/` tồn tại đủ 4 file (`train.csv`, `test.csv`, `features.csv`, `stores.csv`) với đúng số dòng kỳ vọng (421.570 / 115.064 / 8.190 / 45) — cảnh báo sớm nếu ai đó vô tình dùng nhầm bản data khác.
4. Chạy thử `pytest tests/ -k "smoke"` (tập test tối thiểu, không cần train model thật) để xác nhận package đã cài đặt đúng, không chỉ import được mà còn hoạt động.

---

## 5. Quản lý phiên bản khóa cứng (lock) để tái lập chính xác 100%

Khoảng version trong `pyproject.toml` đủ ổn định cho phát triển hàng ngày, nhưng **trước khi nộp bài / peer review**, team nên xuất một file lock để đảm bảo mentor chấm bài chạy đúng environment team đã test:

```bash
pip freeze > environment_lock_YYYYMMDD.txt
```

Lưu file này vào `docs/env_locks/` kèm ngày tạo, không commit đè lên các lần trước — dùng để tra cứu khi có báo lỗi môi trường từ mentor, không dùng để cài đặt hàng ngày (tránh việc mọi người bị khóa cứng version khi đang phát triển).

---

## 6. Cấu trúc bổ sung vào repo cho phần này

```
scripts/
  check_env.py       # Script xác thực môi trường 1 lệnh (mục 4)
docs/
  env_locks/          # Lưu các bản pip freeze trước mỗi lần nộp bài
```

## 7. Lỗi thường gặp & cách xử lý

| Lỗi | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `Microsoft Visual C++ 14.0 required` (Windows) | pip cố build LightGBM/XGBoost từ source thay vì dùng wheel sẵn | Kiểm tra lại version Python nằm trong 3.10–3.12 và dùng pip ≥23 — wheel sẵn chỉ publish cho các bản này |
| `OSError: libomp.dylib not found` (macOS, LightGBM) | Thiếu OpenMP runtime trên macOS | Cài `brew install libomp` trước khi `pip install lightgbm` |
| Kết quả model khác nhau giữa các máy dù cùng code | Thiếu cố định random seed hoặc version package lệch nhau | Chạy `scripts/check_env.py`, đối chiếu với `environment_lock_*.txt`; xác nhận `src/sales_forecast/utils/seed.py` được gọi ở đầu mọi script |
| `pip install -e .` báo lỗi không tìm thấy package `sales_forecast` | Cấu trúc `src/` chưa khớp `[tool.setuptools.packages.find]` trong `pyproject.toml` | Xác nhận đang đứng ở thư mục gốc repo (chứa `pyproject.toml`) khi chạy lệnh |

---

**Nguồn tham khảo phiên bản package (kiểm tra 18/8/2026):**
- [LightGBM Releases — GitHub](https://github.com/lightgbm-org/LightGBM/releases)
- [LightGBM 4.6.0 Release Notes — ReadTheDocs](https://lightgbm.readthedocs.io/_/downloads/en/stable/pdf/)
