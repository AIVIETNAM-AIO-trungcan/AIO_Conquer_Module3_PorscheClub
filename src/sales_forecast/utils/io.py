"""Ghi/đọc file dùng chung — atomic, không biết gì về khái niệm "run".

Ghi atomic (ghi ra file tạm cùng thư mục rồi os.replace()) để tránh file
half-written nếu tiến trình bị ngắt giữa chừng (Ctrl+C, crash) — quan trọng
nhất cho các file con trỏ (vd. latest_run.txt) vì dashboard luôn đọc chúng
đầu tiên và không được thấy nội dung dở dang.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_text_atomic(path: str | Path, text: str) -> None:
    """Ghi text thuần (vd. latest_run.txt) — atomic, không để lại file .tmp
    nếu tiến trình bị ngắt giữa chừng."""
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def read_text(path: str | Path) -> str | None:
    """Đọc text thuần, trả None nếu file chưa tồn tại (không raise)."""
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip()


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    """Ghi JSON — atomic, indent=2 để đọc bằng tay dễ dàng (manifest.json)."""
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    _atomic_write_bytes(Path(path), payload.encode("utf-8"))


def read_json(path: str | Path) -> dict[str, Any] | None:
    """Đọc JSON, trả None nếu file chưa tồn tại (không raise)."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def append_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Nối thêm dòng vào CSV (mode "a") — không đọc lại file cũ để ghi đè.
    Viết header chỉ khi file chưa tồn tại. Dùng cho run_history.csv."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_parquet_atomic(path: str | Path, df) -> None:
    """Ghi DataFrame ra parquet — atomic qua file tạm + os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
