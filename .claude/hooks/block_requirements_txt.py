#!/usr/bin/env python3
"""PreToolUse hook: chan tao file requirements*.txt song song voi pyproject.toml
(invariant #13, CLAUDE.md muc 4).

Ngoai le: docs/env_locks/*.txt la lock file hop le sinh boi `pip freeze`, khong
phai dependency declaration -> khong chan.
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

    if "docs/env_locks/" in normalized:
        return 0

    filename = normalized.rsplit("/", 1)[-1]
    if re.match(r"^requirements.*\.txt$", filename, re.IGNORECASE):
        print(
            "BI CHAN: khong tao requirements.txt song song (CLAUDE.md invariant #13). "
            "Dependency chi khai bao trong pyproject.toml. "
            f"File bi chan: {file_path}. "
            "Neu can lock version truoc khi nop bai, dung `pip freeze > "
            "docs/env_locks/environment_lock_<ngay>.txt` (xem docs/06_environment_setup.md muc 5).",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
