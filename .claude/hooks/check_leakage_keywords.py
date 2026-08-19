#!/usr/bin/env python3
"""PostToolUse hook: canh bao pattern nghi leakage thoi gian trong
src/sales_forecast/features/** va src/sales_forecast/splitting/**
(invariant #1, #2, CLAUDE.md muc 4).

Day la canh bao (khong chan cung) vi khong the xac dinh chinh xac 100% bang
regex - chi bao hieu de Claude tu doi chieu lai voi as_of_date / temporal_split.
"""
import json
import re
import sys
from pathlib import Path

RAW_IO_PATTERNS = [
    r"pd\.read_csv\s*\(",
    r"pd\.read_parquet\s*\(",
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or ""
    normalized = file_path.replace("\\", "/")

    is_features = "src/sales_forecast/features/" in normalized
    is_splitting = "src/sales_forecast/splitting/" in normalized
    if not (is_features or is_splitting):
        return 0

    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return 0

    warnings = []

    if is_features:
        for pattern in RAW_IO_PATTERNS:
            if re.search(pattern, content):
                warnings.append(
                    "doc file truc tiep trong module feature (bo qua temporal_split) - "
                    "invariant #1: Temporal Split phai chay TRUOC Feature Engineering"
                )
                break

        has_as_of_date_param = re.search(r"def\s+\w+\s*\([^)]*as_of_date", content)
        has_date_filter = re.search(r"\[.*[\"']Date[\"'].*[<>]=?", content) or re.search(
            r"\.query\s*\(.*Date", content
        )
        if has_date_filter and not has_as_of_date_param:
            warnings.append(
                "phat hien loc theo Date nhung khong thay tham so as_of_date trong "
                "chu ky ham - invariant #1: moi feature tai thoi diem t chi duoc dung "
                "du lieu Date <= t-1, can nhan as_of_date tuong minh"
            )

        if re.search(r"def\s+\w+\s*\([^)]*as_of_date[^)]*\)", content) and re.search(
            r"(split_date|train_window|valid_window|test_window)", content
        ):
            warnings.append(
                "ham feature co ve vua cat moc thoi gian vua tinh feature - "
                "invariant #2: tach logic feature khoi logic time-boundary, "
                "khong viet 1 ham lam ca 2 viec"
            )

    if warnings:
        joined = "\n  - ".join(warnings)
        print(
            f"CANH BAO leakage/thiet ke ({file_path}):\n  - {joined}\n"
            "Xem lai CLAUDE.md muc 4 (#1, #2) truoc khi bao cao hoan thanh. "
            "Day chi la canh bao heuristic, co the false positive.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
