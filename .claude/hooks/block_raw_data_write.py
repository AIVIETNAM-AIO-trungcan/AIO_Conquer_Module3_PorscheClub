#!/usr/bin/env python3
"""PreToolUse hook: chan moi hanh dong ghi/sua vao data/raw/ (invariant #7, CLAUDE.md muc 4).

data/raw/ la bat bien, chi doc. Hook nay doc JSON tu stdin (chuan Claude Code hook
protocol), kiem tra tool_input.file_path, va chan (exit code 2) neu duong dan nam
trong data/raw/.
"""
import json
import re
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or ""
    normalized = file_path.replace("\\", "/")

    if re.search(r"(^|/)data/raw/", normalized):
        print(
            "BI CHAN: data/raw/ la bat bien, chi doc (CLAUDE.md invariant #7). "
            f"Khong duoc ghi/sua file: {file_path}. "
            "Neu can du lieu trung gian, ghi vao data/interim/ hoac data/processed/.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
