---
name: pre-commit-check
description: Chay dung bo lenh kiem tra truoc khi bao cao 1 thay doi la "xong" - leakage test, conformal/dashboard test, full suite + coverage - roi doi chieu voi checklist 11 muc o CLAUDE.md muc 7. Dung khi nguoi dung hoi "xong chua", "co the commit chua", hoac truoc khi ket thuc 1 task code.
---

# Pre-commit Check

Muc tieu: khong phai tu nho lai 11 muc checklist moi lan — skill nay chay dung thu tu
lenh o CLAUDE.md muc 6 va doi chieu tung muc trong checklist muc 7, tra ve danh sach
pass/fail ro rang.

## Thu tu chay lenh (dung nhu CLAUDE.md muc 6)

1. **Nhom chong leakage truoc tien** (quan trong nhat, chay truoc MOI commit lien quan
   feature/split):
   ```bash
   pytest tests/test_splitting tests/test_features -k "leak" -v
   ```

2. **Nhom conformal prediction** (neu co dung toi `evaluation/conformal.py`):
   ```bash
   pytest tests/test_evaluation -k "conformal" -v
   ```

3. **Nhom dashboard** (neu co dung toi `app/`):
   ```bash
   pytest tests/test_app -v
   ```

4. **Toan bo test suite**:
   ```bash
   pytest tests/ -v
   ```

5. **Coverage** (bat buoc truoc khi merge):
   ```bash
   pytest tests/ --cov=src/sales_forecast --cov-report=term-missing
   ```
   Nguong toi thieu de xuat (`docs/05_test_plan.md` muc 3): >=80% cho
   `ingestion/splitting/features/evaluation/conformal.py`; >=60% cho `models/explainability`;
   >=50% cho `app/`.

Neu buoc 1 fail — DUNG lai, bao cao ngay, khong chay tiep cac buoc sau (leakage la loai
loi nghiem trong nhat trong du an nay).

## Doi chieu checklist CLAUDE.md muc 7 (11 muc)

Sau khi test pass, tu doi chieu tung muc — bao cao ro rang muc nao da xac nhan, muc nao
khong ap dung cho thay doi nay:

- [ ] Da doc `docs/00_decisions.md`, khong vi pham quyet dinh da chot (horizon, chien luoc).
- [ ] Khong co code nao tinh feature tu du lieu tuong lai so voi `as_of_date`.
- [ ] Da them/cap nhat test case tuong ung, `pytest` pass (xem buoc tren).
- [ ] Neu them gia dinh moi -> da cap nhat bang o `docs/05_test_plan.md`.
- [ ] Neu thay doi quyet dinh kien truc -> da cap nhat `docs/00_decisions.md` va giai thich ly do.
- [ ] Logic nghiep vu nam trong `src/`, khong nam trong `pipelines/` hay `notebooks/`.
- [ ] Khong hard-code tham so (horizon, duong dan, search space) — dua vao `configs/`.
- [ ] Comment/docstring giai thich y tuong bang tieng Viet.
- [ ] Neu dung vao khoang tin cay: `calib_window` khong trung/khong dung truoc `train_window`,
      khong dung `test_window`.
- [ ] Neu dung vao `app/`: khong co loi goi train model nao trong code dashboard.
- [ ] Neu them/doi dependency: da cap nhat `pyproject.toml`, KHONG tao `requirements.txt`.

## Bao cao ket qua

Tra ve dang danh sach: `[PASS]`/`[FAIL]`/`[N/A]` cho tung nhom test va tung muc checklist,
kem so dong test pass/fail va % coverage cho cac module lien quan truc tiep den thay doi.
Neu co muc FAIL, KHONG bao cao task la "xong" voi nguoi dung.
