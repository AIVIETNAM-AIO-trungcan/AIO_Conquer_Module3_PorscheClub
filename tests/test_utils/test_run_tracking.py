"""Test cho src/sales_forecast/utils/run_tracking.py — cơ chế run tracking/versioning.

Ánh xạ giả định A25-A33, xem docs/05_test_plan.md.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sales_forecast.utils.io import read_json
from sales_forecast.utils.run_tracking import (
    get_latest_run_id,
    list_runs,
    start_run,
)


@pytest.fixture
def run_dirs(tmp_path):
    """reports/ và data/predictions/ tạm, không đụng dữ liệu thật."""
    reports = tmp_path / "reports"
    predictions = tmp_path / "data" / "predictions"
    return reports, predictions


def test_run_id_never_collides_same_second(run_dirs, monkeypatch):
    """A25: 2 lần gọi start_run() liên tiếp cùng pipeline_name (kể cả cùng
    giây) phải sinh run_id khác nhau, không ghi đè lên nhau."""
    reports, predictions = run_dirs
    ctx1 = start_run(reports, predictions, "train_baseline")
    ctx1.reports_path("figures", "decision_tree.png").write_text("a")
    ctx1.finalize(status="success")

    ctx2 = start_run(reports, predictions, "train_baseline")
    assert ctx2.run_id != ctx1.run_id


def test_run_context_never_overwrites_previous_run(run_dirs):
    """A26: ghi file qua RunContext của run sau không đụng file của run trước."""
    reports, predictions = run_dirs
    ctx1 = start_run(reports, predictions, "train_baseline")
    f1 = ctx1.reports_path("figures", "decision_tree.png")
    f1.write_text("run1")
    ctx1.finalize(status="success")

    ctx2 = start_run(reports, predictions, "train_baseline")
    f2 = ctx2.reports_path("figures", "decision_tree.png")
    f2.write_text("run2")
    ctx2.finalize(status="success")

    assert f1 != f2
    assert f1.read_text() == "run1"
    assert f2.read_text() == "run2"


def test_latest_pointer_matches_most_recent_successful_run(run_dirs):
    """A27: sau finalize(status="success"), get_latest_run_id() trả đúng run
    vừa hoàn tất."""
    reports, predictions = run_dirs
    ctx1 = start_run(reports, predictions, "train_baseline")
    ctx1.reports_path("figures", "decision_tree.png").write_text("a")
    ctx1.finalize(status="success")
    assert get_latest_run_id(reports) == ctx1.run_id

    ctx2 = start_run(reports, predictions, "train_baseline")
    ctx2.reports_path("figures", "decision_tree.png").write_text("b")
    ctx2.finalize(status="success")
    assert get_latest_run_id(reports) == ctx2.run_id


def test_failed_run_does_not_update_latest_pointer(run_dirs):
    """A28: run lỗi (status="failed") không được làm latest_run.txt trỏ vào
    dữ liệu thiếu/dở dang — pointer vẫn giữ nguyên run thành công gần nhất."""
    reports, predictions = run_dirs
    ctx1 = start_run(reports, predictions, "train_baseline")
    ctx1.reports_path("figures", "decision_tree.png").write_text("a")
    ctx1.finalize(status="success")

    ctx2 = start_run(reports, predictions, "train_baseline")
    ctx2.reports_path("figures", "decision_tree.png").write_text("b (incomplete)")
    ctx2.finalize(status="failed")

    assert get_latest_run_id(reports) == ctx1.run_id


def test_latest_pointers_independent_per_base_dir(run_dirs):
    """A29: reports/latest_run.txt và data/predictions/latest_run.txt độc lập
    — 1 run chỉ ghi vào reports/ không được đụng pointer của data/predictions/."""
    reports, predictions = run_dirs

    ctx1 = start_run(reports, predictions, "train_baseline")
    ctx1.predictions_path("predictions_long.parquet").write_text("x")
    ctx1.finalize(status="success")
    assert get_latest_run_id(predictions) == ctx1.run_id
    assert get_latest_run_id(reports) is None  # chưa từng ghi vào reports/

    ctx2 = start_run(reports, predictions, "run_shap")
    ctx2.reports_path("shap_summary.png").write_text("y")
    ctx2.finalize(status="success")

    assert get_latest_run_id(reports) == ctx2.run_id
    # run_shap không ghi predictions -> pointer predictions/ vẫn giữ nguyên ctx1
    assert get_latest_run_id(predictions) == ctx1.run_id


def test_manifest_contains_required_fields(run_dirs):
    """A30: manifest.json chứa đủ trường bắt buộc sau finalize()."""
    reports, predictions = run_dirs
    ctx = start_run(reports, predictions, "train_baseline")
    ctx.write_manifest(
        {
            "config_snapshot": {"data": {"raw_dir": "data/raw"}},
            "metrics_summary": {"naive": {"wmae": 100.0, "wmape": 0.1}},
        }
    )
    ctx.finalize(status="success")

    manifest_path = reports / "runs" / ctx.run_id / "manifest.json"
    manifest = read_json(manifest_path)
    assert manifest is not None
    for key in ("run_id", "pipeline_name", "started_at", "finished_at", "status"):
        assert key in manifest
    assert manifest["run_id"] == ctx.run_id
    assert manifest["status"] == "success"
    assert manifest["finished_at"] is not None
    assert "config_snapshot" in manifest
    assert "metrics_summary" in manifest


def test_run_history_append_only_preserves_old_rows(run_dirs):
    """A31: append_run_history() không đọc/ghi đè các dòng cũ, chỉ nối thêm."""
    reports, predictions = run_dirs
    ctx1 = start_run(reports, predictions, "train_baseline")
    ctx1.append_run_history([{"model_name": "naive", "wmae": 100.0, "wmape": 0.1}])
    ctx1.finalize(status="success")

    ctx2 = start_run(reports, predictions, "train_baseline")
    ctx2.append_run_history(
        [
            {"model_name": "naive", "wmae": 95.0, "wmape": 0.09},
            {"model_name": "decision_tree", "wmae": 80.0, "wmape": 0.08},
        ]
    )
    ctx2.finalize(status="success")

    history = pd.read_csv(reports / "run_history.csv")
    assert len(history) == 3
    assert set(history["run_id"]) == {ctx1.run_id, ctx2.run_id}
    assert list(history[history["run_id"] == ctx1.run_id]["wmae"]) == [100.0]


def test_list_runs_sorted_and_filters_invalid_dirs(run_dirs):
    """A32: list_runs() trả về đúng thứ tự giảm dần theo thời gian, bỏ qua
    thư mục không đúng format run_id."""
    reports, predictions = run_dirs
    ctx1 = start_run(reports, predictions, "train_baseline")
    ctx1.reports_path("figures", "x.png").write_text("a")
    ctx1.finalize(status="success")

    ctx2 = start_run(reports, predictions, "train_baseline")
    ctx2.reports_path("figures", "x.png").write_text("b")
    ctx2.finalize(status="success")

    # thư mục rác không đúng format run_id
    junk_dir = reports / "runs" / "not_a_run_id"
    junk_dir.mkdir(parents=True)

    runs = list_runs(reports)
    assert runs[0] == max(ctx1.run_id, ctx2.run_id)
    assert "not_a_run_id" not in runs
    assert set(runs) == {ctx1.run_id, ctx2.run_id}


def test_atomic_write_leaves_no_partial_file(run_dirs):
    """A33: ghi manifest.json/latest_run.txt không để lại file .tmp sau khi
    ghi thành công."""
    reports, predictions = run_dirs
    ctx = start_run(reports, predictions, "train_baseline")
    ctx.reports_path("figures", "x.png").write_text("a")
    ctx.write_manifest({"config_snapshot": {}, "metrics_summary": {}})
    ctx.finalize(status="success")

    run_dir = reports / "runs" / ctx.run_id
    tmp_files = list(run_dir.glob("*.tmp")) + list(reports.glob("*.tmp"))
    assert tmp_files == []
