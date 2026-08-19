---
name: add-model
description: Quy trinh chuan them hoac sua 1 model trong src/sales_forecast/models/registry.py - dam bao dung interface .fit/.predict thong nhat, khong dung logic evaluation rieng, cap nhat dung config. Dung khi nguoi dung yeu cau "them model moi", "sua wrapper model X", hoac "dang ky model vao registry".
---

# Add Model

Dam bao invariant #4 (CLAUDE.md muc 4): mot Evaluation Layer dung chung cho MOI model,
khong viet logic danh gia rieng le cho tung model.

## Buoc 0 - Kiem tra pham vi model da chot

Doc `docs/00_decisions.md` bang tom tat, dong "Model tree-based bo sung" — hien tai
CHI 5 model bat buoc Module 03 (Decision Tree, Random Forest, AdaBoost, Gradient Boost,
LightGBM/XGBoost), KHONG dung CatBoost (da bi loai bo tuong minh). Neu yeu cau them mot
model ngoai danh sach nay, dung lai va xac nhan voi nguoi dung truoc khi code — day la
thay doi quyet dinh kien truc da chot, can nguoi dung xac nhan ly do (theo CLAUDE.md muc 2).

## Buoc 1 - Doc interface hien co

Doc `src/sales_forecast/models/registry.py` va it nhat 1 wrapper da co (vd.
`tree_models.py` hoac `boosting_models.py`) de lay dung chu ky `.fit(X, y)` /
`.predict(X)` hien tai — KHONG doan.

## Buoc 2 - Viet/sua wrapper

- Model (ke ca baseline) PHAI expose dung `.fit(X, y)` va `.predict(X)`, khong if/else
  rieng theo loai model o noi goi (invariant #4, test A12).
- KHONG import hay goi bat cu ham nao trong `src/sales_forecast/evaluation/` de tu
  danh gia rieng — Evaluation Layer se goi model tu ben ngoai qua registry.
- Dang ky model trong `models/registry.py` theo dung co che dat ten hien co
  (vd. `get_model("lightgbm")`).

## Buoc 3 - Cau hinh

Them/sua `configs/model_<ten>.yaml` tuong ung — tham so model (n_estimators, max_depth,
learning_rate...) khong hard-code trong `src/`. Neu model tham gia Optuna tuning, kiem tra
search space tuong ung trong `configs/optuna.yaml`.

**Luu y horizon**: theo `docs/00_decisions.md`, chien luoc la Direct multi-step chia 3
nhom horizon (h=1-4, h=5-12, h=13-39) — moi nhom co the can model/feature set rieng.
Xac nhan xem model moi co can train rieng theo nhom horizon khong truoc khi code.

## Buoc 4 - Test

Cap nhat `tests/test_models/test_model_interface_consistency.py` — them ten model moi
vao `@pytest.mark.parametrize` (mapping A12, xem skeleton co san trong
`docs/05_test_plan.md`). Neu model duoc train qua Optuna, dam bao khong vi pham A13
(Optuna khong duoc thay `test_window`).

## Xac nhan cuoi

```bash
pytest tests/test_models -v
```
Kiem tra checklist CLAUDE.md muc 7 truoc khi bao cao hoan thanh — dac biet muc
"Logic nghiep vu nam trong src/" va "Khong hard-code tham so".
