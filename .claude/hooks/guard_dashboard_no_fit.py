#!/usr/bin/env python3
"""PostToolUse hook: canh bao neu code trong app/ goi .fit(/.train( cua model
(invariant #12, CLAUDE.md muc 4 - dashboard la lop trinh bay thuan tuy).

Day la canh bao (khong chan cung) vi co the false positive voi method trung ten
(vd. mot class UI tu viet co .fit() rieng). In canh bao ro rang de Claude tu xac
nhan lai truoc khi bao "xong".
"""
import json
import re
import sys
from pathlib import Path

SUSPECT_PATTERNS = [
    r"\.fit\s*\(",
    r"\.train\s*\(",
    r"lgb\.train",
    r"xgboost\.train",
    r"XGBRegressor\s*\(",
    r"LGBMRegressor\s*\(",
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or ""
    normalized = file_path.replace("\\", "/")

    if "app/" not in normalized:
        return 0

    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return 0

    hits = []
    for pattern in SUSPECT_PATTERNS:
        if re.search(pattern, content):
            hits.append(pattern)

    if hits:
        print(
            "CANH BAO: phat hien pattern nghi train model trong app/ "
            f"({file_path}): {', '.join(hits)}. "
            "Invariant #12 (CLAUDE.md muc 4): dashboard KHONG duoc goi .fit()/.train() "
            "cua bat ky model nao, chi doc file da co trong reports/ va data/predictions/. "
            "Neu day la false positive (vd. method tu dinh nghia khac ngu nghia), "
            "xac nhan lai truoc khi bao cao hoan thanh.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
