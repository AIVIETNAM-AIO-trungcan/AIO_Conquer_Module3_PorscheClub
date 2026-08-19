---
name: new-feature-block
description: Quy trinh chuan them 1 feature block moi vao src/sales_forecast/features/ (giai doan 3 pipeline) - dam bao dung interface FeatureBlock, khong leakage thoi gian, co config bat/tat, co test tuong ung. Dung khi nguoi dung yeu cau "them feature", "viet feature block moi", hoac "them cot X vao feature_matrix".
---

# New Feature Block

Quy trinh nay dam bao moi feature block moi tuan thu invariant #1, #2, #3, #9 (CLAUDE.md muc 4)
va khong bo sot buoc nao trong 3 buoc bat buoc: code, config, test.

## Buoc 0 - Doc truoc khi viet

Doc `src/sales_forecast/features/base.py` de lay dung interface `FeatureBlock` hien co
(khong doan chu ky ham). Doc 1 block da co (vd. `calendar.py` hoac `lag_rolling.py`) lam
mau ve style va cach nhan `as_of_date`.

Kiem tra `docs/00_decisions.md` xem feature nay co lien quan quyet dinh da chot nao khong
(vd. neu feature phan biet theo nhom horizon h=1-4/5-12/13-39, xem muc "Forecast horizon").

## Buoc 1 - Viet ham thuan trong `src/sales_forecast/features/<ten_block>.py`

- Ham chinh phai co dang `f(df, as_of_date) -> df_features`, KHONG side-effect.
- KHONG doc file truc tiep (`pd.read_csv`, `pd.read_parquet`) trong module nay - nhan
  DataFrame da duoc load/split tu ben ngoai (invariant #2: tach logic feature khoi
  logic time-boundary).
- Moi dong voi `Date = t` chi duoc dung du lieu `Date <= as_of_date - 1` (invariant #1).
- Neu feature co the NaN co y nghia nghiep vu (vd. chua co lich su, khong co khuyen mai),
  PHAI co cot flag tuong minh (vd. `has_history`, `has_markdown`) thay vi fillna mu quang
  (invariant #9). Tham khao `docs/00_decisions.md` muc "Xu ly MarkDown missing" va
  "Xu ly cold-start" cho 2 vi du da co san.
- Ke thua/tuan thu dung interface `FeatureBlock` da doc o Buoc 0.

## Buoc 2 - Dang ky trong `configs/features.yaml`

Them entry bat/tat cho block moi (invariant #3 - khong hard-code trong `src/`). Tham so
lag/rolling window/tham so rieng cua block cung phai nam trong config nay, khong hard-code.

Sau khi them, cap nhat `src/sales_forecast/features/pipeline.py` de doc entry moi tu config
va ghep block vao pipeline theo dung co che da co (xem cac block khac lam mau).

## Buoc 3 - Viet test

Theo dung tinh than invariant #6: moi gia dinh quan trong phai co test. Toi thieu 2 loai test:

1. **Test khong leakage** (mapping A6 trong `docs/05_test_plan.md`): dong voi `Date = t`
   chi duoc phan anh du lieu `<= t-1`; dong dau tien cua 1 chuoi phai co NaN o cac cot
   lag-like, KHONG duoc dien 0.
2. **Test doc lap block** (mapping A9): tat block nay qua config khong duoc lam hong cac
   block khac; cac cot cua block khac van con nguyen ven.

Dat file test doi xung: `tests/test_features/test_<ten_block>.py`.

Neu feature co flag NaN-co-y-nghia (Buoc 1), them test rieng kiem tra flag dung
(mau tham khao: `tests/test_features/test_markdown_flag.py`).

## Buoc 4 - Them dong vao bang mapping

Neu day la mot gia dinh MOI chua co trong bang o `docs/05_test_plan.md` muc 1, phai them
1 dong moi vao bang do truoc khi coi task la xong (invariant #6). Dung skill
`update-decision` neu day cung la 1 quyet dinh kien truc can ghi vao `docs/00_decisions.md`.

## Xac nhan cuoi

Chay:
```bash
pytest tests/test_features/test_<ten_block>.py -v
pytest tests/test_features -k "leak" -v
```
Ca hai phai pass truoc khi bao cao hoan thanh.
