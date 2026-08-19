#!/usr/bin/env python3
"""UserPromptSubmit hook: khi prompt nguoi dung nhac toi horizon/chien luoc/split,
tu dong chen 1 ban tom tat NGAN cac quyet dinh da chot (thay vi de Claude tu Read
toan bo docs/00_decisions.md dai ~130 dong moi lan - day la phan toi uu token
truc tiep cua framework nay).

Ban tom tat duoc GIU CO DINH tai day (khong doc file dong), vi noi dung quyet
dinh thay doi hiem (chi khi team hop lai). Neu docs/00_decisions.md duoc cap
nhat, phai cap nhat lai ban tom tat nay theo (nhac trong output).
"""
import json
import sys

TRIGGER_KEYWORDS = [
    "horizon", "recursive", "direct", "hybrid", "chien luoc", "chiến lược",
    "temporal split", "as_of_date", "calib_window", "train_window",
    "valid_window", "test_window", "conformal",
]

SUMMARY = """\
[Nhac tu hook remind_decisions_check - tom tat docs/00_decisions.md, KHONG can Read lai file]
Cac quyet dinh kien truc DA CHOT (2026-08-18, trang thai: DA CHOT TOAN BO):
- Don vi du bao: (Store, Dept, Date), tan suat tuan.
- Horizon & chien luoc: Direct multi-step (KHONG recursive), horizon = 39 tuan,
  chia 3 nhom h=1-4, h=5-12, h=13-39.
- Metric: bao cao SONG SONG ca WMAE va WMAPE, trong so IsHoliday x5.
- Weekly_Sales am: giu nguyen, KHONG clip ve 0.
- MarkDown NaN: flag has_markdown tuong minh, KHONG fillna(0) mu quang.
- Cold-start (11 cap Store-Dept): flag has_history=False + fallback trung binh
  theo Dept/Store Type; do rieng metric nhom cold-start o Giai doan 7.
- Khoang tin cay 95%: Split Conformal Prediction. calib_window tach rieng tu
  valid_window (50/50 voi phan Optuna), LUON nam SAU train_window, KHONG BAO GIO
  dung test_window de tinh residual hieu chinh.
- Model: chi 5 model bat buoc Module 03 (Decision Tree, RF, AdaBoost, Gradient
  Boost, LightGBM/XGBoost). KHONG dung CatBoost (da loai bo).
- Moi truong: venv + pip + pyproject.toml.
- Dashboard: Streamlit, doc reports/ + data/predictions/, KHONG train model.

Neu can chi tiet ly do/bang chung day du cho 1 muc cu the, Read truc tiep
docs/00_decisions.md muc tuong ung thay vi doc toan bo file.
"""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    prompt = (payload.get("prompt") or "").lower()
    if any(keyword in prompt for keyword in TRIGGER_KEYWORDS):
        print(SUMMARY)

    return 0


if __name__ == "__main__":
    sys.exit(main())
