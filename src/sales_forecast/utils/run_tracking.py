"""Run tracking — versioning output của mỗi lần chạy pipeline.

Mỗi lần chạy pipeline sinh 1 run_id riêng, output ghi vào thư mục con
reports/runs/<run_id>/ và data/predictions/runs/<run_id>/ thay vì ghi đè
lên path cố định — cho phép so sánh WMAE/WMAPE/coverage giữa các lần chạy
(vd. trước/sau thêm feature, trước/sau Optuna tuning) qua reports/run_history.csv.

Không dùng MLflow/DVC: tránh thêm dependency nặng, giữ đúng triết lý
venv+pip đơn giản đã chốt ở docs/06_environment_setup.md — xem lý do đầy đủ
ở docs/00_decisions.md.

Retention: module này KHÔNG tự động xoá run cũ. Ở quy mô đồ án học thuật,
số lần chạy hữu hạn (vài chục lần là nhiều); xoá tự động có rủi ro mất
bằng chứng cần nộp bài. Dọn thủ công qua scripts/prune_old_runs.py nếu cần.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sales_forecast.utils.io import (
    append_csv_rows,
    read_text,
    write_json_atomic,
    write_text_atomic,
)

RUN_ID_PATTERN = re.compile(r"^.+_\d{8}_\d{6}(?:_\d+)?$")
RUN_HISTORY_FIELDS = [
    "run_id",
    "started_at",
    "pipeline_name",
    "model_name",
    "wmae",
    "wmape",
    "coverage_95",
    "status",
]


def _runs_dir(base_dir: Path) -> Path:
    return base_dir / "runs"


def _latest_pointer_path(base_dir: Path) -> Path:
    return base_dir / "latest_run.txt"


def _generate_run_id(reports_dir: Path, pipeline_name: str, now: datetime) -> str:
    """Sinh run_id "{pipeline_name}_{YYYYMMDD_HHMMSS}", thêm suffix _2/_3...
    nếu thư mục run đã tồn tại (2 lần chạy trùng giây, cùng pipeline) — không
    bao giờ ghi đè âm thầm lên run đã có."""
    base_id = f"{pipeline_name}_{now.strftime('%Y%m%d_%H%M%S')}"
    candidate = base_id
    suffix = 1
    existing = _runs_dir(reports_dir)
    while (existing / candidate).exists():
        suffix += 1
        candidate = f"{base_id}_{suffix}"
    return candidate


@dataclass
class RunContext:
    """Đại diện 1 lần chạy pipeline — mọi output của run này đi qua đây."""

    run_id: str
    pipeline_name: str
    started_at: datetime
    reports_base: Path
    predictions_base: Path
    _touched_reports: bool = field(default=False, init=False)
    _touched_predictions: bool = field(default=False, init=False)

    def reports_path(self, *parts: str) -> Path:
        """reports/runs/<run_id>/<parts>, tự mkdir thư mục cha."""
        self._touched_reports = True
        p = _runs_dir(self.reports_base) / self.run_id / Path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def predictions_path(self, *parts: str) -> Path:
        """data/predictions/runs/<run_id>/<parts>, tự mkdir thư mục cha."""
        self._touched_predictions = True
        p = _runs_dir(self.predictions_base) / self.run_id / Path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _manifest_path(self) -> Path:
        return _runs_dir(self.reports_base) / self.run_id / "manifest.json"

    def write_manifest(self, extra: dict[str, Any]) -> None:
        """Ghi reports/runs/<run_id>/manifest.json (atomic). `extra` thường
        chứa config_snapshot và metrics_summary — gọi trước finalize()."""
        self._touched_reports = True
        manifest = {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "started_at": self.started_at.isoformat(),
            "finished_at": None,
            "status": "running",
            **extra,
        }
        write_json_atomic(self._manifest_path(), manifest)

    def append_run_history(self, rows: list[dict[str, Any]]) -> None:
        """Nối thêm N dòng (1 dòng/model) vào reports/run_history.csv."""
        full_rows = []
        for row in rows:
            full_rows.append(
                {
                    "run_id": self.run_id,
                    "started_at": self.started_at.isoformat(),
                    "pipeline_name": self.pipeline_name,
                    "coverage_95": row.get("coverage_95", ""),
                    "status": row.get("status", "success"),
                    **{k: v for k, v in row.items() if k in ("model_name", "wmae", "wmape")},
                }
            )
        append_csv_rows(self.reports_base / "run_history.csv", full_rows, RUN_HISTORY_FIELDS)

    def finalize(self, status: Literal["success", "failed"]) -> None:
        """Cập nhật manifest (status/finished_at). Chỉ khi status="success":
        cập nhật latest_run.txt của reports/ (nếu _touched_reports) và/hoặc
        data/predictions/ (nếu _touched_predictions). Gọi trong finally-block
        để luôn chạy kể cả khi pipeline lỗi giữa chừng — nhờ vậy 1 run lỗi/dở
        dang KHÔNG BAO GIỜ làm latest pointer trỏ vào dữ liệu thiếu."""
        manifest_path = self._manifest_path()
        if manifest_path.exists():
            from sales_forecast.utils.io import read_json

            manifest = read_json(manifest_path) or {}
        else:
            manifest = {
                "run_id": self.run_id,
                "pipeline_name": self.pipeline_name,
                "started_at": self.started_at.isoformat(),
            }
        manifest["status"] = status
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)

        if status != "success":
            return
        if self._touched_reports:
            write_text_atomic(_latest_pointer_path(self.reports_base), self.run_id)
        if self._touched_predictions:
            write_text_atomic(_latest_pointer_path(self.predictions_base), self.run_id)


def start_run(reports_base: Path, predictions_base: Path, pipeline_name: str) -> RunContext:
    """Sinh run_id mới, trả về RunContext. KHÔNG tạo thư mục run ngay — thư
    mục con (figures/, metrics/...) chỉ tạo khi thực sự có file ghi vào qua
    reports_path()/predictions_path(), tránh thư mục rỗng nếu pipeline lỗi
    trước khi ghi được gì."""
    now = datetime.now(timezone.utc)
    run_id = _generate_run_id(reports_base, pipeline_name, now)
    return RunContext(
        run_id=run_id,
        pipeline_name=pipeline_name,
        started_at=now,
        reports_base=reports_base,
        predictions_base=predictions_base,
    )


def get_latest_run_id(base_dir: Path) -> str | None:
    """Đọc <base_dir>/latest_run.txt, trả None nếu chưa có run thành công nào.
    base_dir là reports/ hoặc data/predictions/ — 2 pointer độc lập."""
    return read_text(_latest_pointer_path(base_dir))


def list_runs(base_dir: Path) -> list[str]:
    """Liệt kê run_id trong <base_dir>/runs/, sort giảm dần theo timestamp
    trong tên (mới nhất trước), bỏ qua thư mục không đúng format run_id."""
    runs_dir = _runs_dir(base_dir)
    if not runs_dir.exists():
        return []
    run_ids = [p.name for p in runs_dir.iterdir() if p.is_dir() and RUN_ID_PATTERN.match(p.name)]
    return sorted(run_ids, reverse=True)
