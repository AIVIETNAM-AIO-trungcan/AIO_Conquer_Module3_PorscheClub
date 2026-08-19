---
name: conformal-safety-check
description: Kiem tra 3 rang buoc thoi gian bat buoc cua Split Conformal Prediction (calib_window tach khoi train_window, calib sau train, khong dung test_window) truoc khi coi 1 thay doi trong evaluation/conformal.py la xong. Dung khi nguoi dung dung toi calib_window, conformal calibration, hoac khoang tin cay 95%.
---

# Conformal Safety Check

Invariant #11 (CLAUDE.md muc 4) va chi tiet day du o `docs/08_uncertainty_conformal.md`
muc 3 — day la diem de sai nhat khi them conformal prediction vao pipeline da co san
co che chong leakage. Skill nay bat buoc doi chieu code voi 3 rang buoc, khong chi dua
vao ghi nho.

## Rang buoc 1 - `calib_window` tach rieng khoi `train_window`

Doc code vua sua trong `evaluation/conformal.py` (va noi goi no, thuong o
`pipelines/run_conformal_calibration.py`). Xac nhan:
- Residual `r_i = |y_i - y_hat_i|` dung de tinh quantile calibration PHAI lay tu
  `calib_window`, KHONG duoc lay tu chinh `train_window` da dung de fit model
  (neu dung train_window, residual se bi danh gia thap he thong, khoang tin cay qua hep).

## Rang buoc 2 - `calib_window` nam SAU `train_window` ve thoi gian

- Khong duoc lay ngau nhien tu giua chuoi thoi gian — phai theo dung nguyen tac
  walk-forward da co o Temporal Split (giai doan 2).
- Theo `docs/00_decisions.md`: `calib_window` la nua sau cua `valid_window`, tach 50/50
  voi phan dung cho Optuna (nua dau). Xac nhan code chia dung ty le va dung thu tu
  thoi gian (khong phai chia ngau nhien 50/50).

## Rang buoc 3 - KHONG BAO GIO dung `test_window` de tinh residual hieu chinh

- Grep code vua sua tim bien `test_window` trong pham vi ham tinh calibrator — neu co
  xuat hien, DUNG lai va bao cao ngay, day la vi pham nghiem trong nhat trong 3 rang buoc.

## Kiem tra cong thuc (A18 trong docs/05_test_plan.md)

Xac nhan quantile dung cong thuc hieu chinh huu han mau:
```
q = quantile_{ceil((n+1)(1-alpha)) / n} ({r_1, ..., r_n})
```
KHONG phai `np.quantile(residuals, 0.95)` tran (sai voi n nho). Doi chieu voi
`tests/test_evaluation/test_conformal_prediction.py::test_conformal_finite_sample_correction`.

## Sau khi kiem tra code — chay test

```bash
pytest tests/test_evaluation -k "conformal" -v
```

Xac nhan ca 5 test A16-A20 pass (xem bang mapping day du o `docs/05_test_plan.md` va
`docs/08_uncertainty_conformal.md` muc 6):
- A16: khong overlap train/calib
- A17: calib sau train ve thoi gian
- A18: hieu chinh huu han mau dung cong thuc
- A19: `y_lo <= y_pred <= y_hi` moi dong
- A20: empirical coverage tren valid_window nam trong khoang chap nhan duoc quanh 95%

## Luu y bao cao

Theo `docs/08_uncertainty_conformal.md` muc 5, khi bao cao ket qua conformal KHONG chi
bao cao 1 con so coverage trung binh — phai bao gom ca Average Interval Width va coverage
theo horizon/cold-start. Neu task hien tai co sinh report, xac nhan cac lat cat nay da co.
