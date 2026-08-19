---
name: update-decision
description: Quy trinh ghi 1 quyet dinh kien truc moi vao docs/00_decisions.md dung template, kem kiem tra xem co can them dong vao docs/05_test_plan.md khong. Dung khi nguoi dung chot mot lua chon kien truc moi (horizon, metric, xu ly missing, model, moi truong...) can luu lai.
---

# Update Decision

Invariant #10 (CLAUDE.md muc 4): ghi quyet dinh kien truc quan trong vao
`docs/00_decisions.md` ngay khi chot — khong de quyet dinh chi ton tai trong chat hoac
comment code rai rac.

## Buoc 1 - Xac nhan day thuc su la quyet dinh kien truc

Phan biet: 1 lua chon tam thoi trong luc code (khong can ghi) vs. 1 quyet dinh anh huong
nhieu module / co the gay tranh cai lai sau nay (can ghi). Neu khong chac, hoi nguoi dung
truoc khi ghi — dung tu y coi moi cau noi la 1 "quyet dinh chinh thuc".

Neu quyet dinh nay THAY DOI mot quyet dinh da chot truoc do trong `docs/00_decisions.md`
(vd. dao nguoc lua chon horizon, doi metric chinh) — theo CLAUDE.md muc 2, phai neu ro
ly do va xin xac nhan cua nguoi dung TRUOC khi ghi, khong tu y sua doi kien truc da chot.

## Buoc 2 - Doc "Mau them quyet dinh moi" o cuoi file

Doc phan cuoi cung cua `docs/00_decisions.md` (muc "Mau them quyet dinh moi") de lay dung
template hien hanh, khong tu bia format.

## Buoc 3 - Sinh block markdown moi

Them 1 muc moi (KHONG xoa quyet dinh cu — giu lai lich su theo dung nguyen tac cua file):

```markdown
## [YYYY-MM-DD] Ten quyet dinh

**Boi canh:** ...
**Cac lua chon da xem xet:** ...
**Quyet dinh:** ...
**Ly do:** ...
**Anh huong toi cac module:** liet ke ro file/thu muc bi anh huong
**Test case lien quan (neu co):** ...
```

Ngay thang dung dinh dang tuyet doi (YYYY-MM-DD), khong dung tu tuong doi ("hom nay",
"tuan sau").

Neu quyet dinh nay thay doi bang tom tat o dau file `docs/00_decisions.md`, cap nhat luon
dong tuong ung trong bang do.

## Buoc 4 - Kiem tra can them test khong

Doi chieu voi invariant #6: neu quyet dinh nay tao ra 1 gia dinh moi co the kiem tra duoc
(vd. "calib_window phai 50/50 voi phan Optuna"), phai them 1 dong vao bang mapping o
`docs/05_test_plan.md` muc 1 truoc khi coi task la xong — dung ky hieu tiep theo (A25,
A26...) va tham chieu file test se viet.

## Buoc 5 - Ra soat module bi anh huong

Voi moi module liet ke o "Anh huong toi cac module", xac nhan code hien tai (neu da co)
co dang di nguoc quyet dinh moi khong — neu co, bao cho nguoi dung biet can sua o dau
truoc khi coi quyet dinh la da ap dung xong (ghi tai lieu khac voi ap dung vao code).
