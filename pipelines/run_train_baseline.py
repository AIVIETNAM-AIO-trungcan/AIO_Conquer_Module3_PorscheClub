"""Orchestration: load raw -> validate -> aggregate -> feature matrix ->
split (ty le 2/3) -> fit baseline models -> đo WMAE/WMAPE trên valid_window.

Chỉ orchestration (CLAUDE.md mục 4 rule: pipelines/ không chứa logic nghiệp vụ,
mọi logic thật nằm trong src/sales_forecast/).

Thứ tự Feature Engineering TRƯỚC Split (đảo ngược cục bộ so với invariant #1
CLAUDE.md, CHỈ áp dụng trong pipeline này) — quyết định team, xem
docs/00_decisions.md "Đồng bộ xử lý dữ liệu theo notebooks/01. Preprocessing.ipynb".

Cách dùng:
    python pipelines/run_train_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sales_forecast.evaluation.metrics import weighted_mae, weighted_mape
from sales_forecast.explainability.tree_plot import plot_decision_tree
from sales_forecast.features.macro import apply_macro_lag52_to_valid
from sales_forecast.features.pipeline import (
    build_feature_matrix,
    load_enabled_blocks_from_config,
    load_lag_rolling_params_from_config,
)
from sales_forecast.ingestion.loaders import aggregate_to_store_date, load_raw_data
from sales_forecast.ingestion.validators import (
    validate_features_schema,
    validate_stores_schema,
    validate_test_aggregated_schema,
    validate_test_schema,
    validate_train_aggregated_schema,
    validate_train_schema,
)
from sales_forecast.models.baseline import NaiveSameWeekLastYear, SimpleDecisionTreeBaseline
from sales_forecast.splitting.ratio_split import split_by_date_ratio
from sales_forecast.utils.run_tracking import start_run


def main() -> int:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    baseline_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "model_baseline.yaml").read_text(encoding="utf-8")
    )

    run_ctx = start_run(
        reports_base=REPO_ROOT / data_cfg["data_paths"]["reports_dir"],
        predictions_base=REPO_ROOT / data_cfg["data_paths"]["predictions_dir"],
        pipeline_name="train_baseline",
    )
    print(f"=== run_id: {run_ctx.run_id} ===")
    try:
        exit_code = _run(run_ctx, data_cfg, baseline_cfg)
    except BaseException:
        run_ctx.finalize(status="failed")
        raise
    run_ctx.finalize(status="success" if exit_code == 0 else "failed")
    return exit_code


def _run(run_ctx, data_cfg: dict, baseline_cfg: dict) -> int:
    print("=== 1. Load & validate raw data ===")
    raw = load_raw_data(REPO_ROOT / data_cfg["data_paths"]["raw_dir"])
    validate_train_schema(raw["train"])
    validate_test_schema(raw["test"])
    validate_features_schema(raw["features"])
    validate_stores_schema(raw["stores"])
    print(f"train: {raw['train'].shape}, features: {raw['features'].shape}")

    print("\n=== 1b. Aggregate (Store, Dept, Date) -> (Store, Date) ===")
    # Đơn vị dự báo đã đổi — xem docs/00_decisions.md [2026-08-19]
    # "Đổi đơn vị dự báo: (Store, Dept, Date) -> (Store, Date)". Aggregate
    # chạy NGAY SAU validate schema raw, TRƯỚC Temporal Split.
    train_agg, test_agg = aggregate_to_store_date(raw["train"], raw["test"])
    validate_train_aggregated_schema(train_agg)
    validate_test_aggregated_schema(test_agg)
    print(f"train (aggregated): {train_agg.shape}, test (aggregated): {test_agg.shape}")

    print("\n=== 2. Feature Engineering (chay TRUOC split - quyet dinh team, xem")
    print("=== docs/00_decisions.md 'Dong bo xu ly du lieu theo notebooks/01. Preprocessing.ipynb') ===")
    # Đảo thứ tự CỤC BỘ trong pipeline này: tính Lag/Rolling/Calendar/MarkDown/Macro
    # trên toàn bộ train_agg + test_agg gộp TRƯỚC khi tách train/valid — khớp
    # notebooks/01. Preprocessing.ipynb Cell 4-10. KHÔNG áp dụng cho kiến trúc
    # chung (CLAUDE.md invariant #1, docs/02_pipeline_architecture.md không đổi).
    enabled_blocks = load_enabled_blocks_from_config(REPO_ROOT / "configs" / "features.yaml")
    lags, rolling_windows = load_lag_rolling_params_from_config(REPO_ROOT / "configs" / "features.yaml")
    print(f"enabled_blocks (tu configs/features.yaml): {enabled_blocks}")
    feature_matrix = build_feature_matrix(
        train_agg, test_agg, raw["features"],
        enabled_blocks=enabled_blocks, lags=lags, rolling_windows=rolling_windows,
    )
    print(f"feature_matrix (truoc dropna lag_52w): {feature_matrix.shape}")

    print("\n=== 3. Temporal Split (theo ty le 2/3 so ngay duy nhat) ===")
    # Chi giu cac dong du lich su lag_52w (khop notebook: dropna(subset=['Lag_52'])
    # truoc khi chia) - loai 52 tuan dau moi Store khong du du lieu lag dai nhat.
    feature_matrix_full_history = feature_matrix.dropna(subset=["lag_52w"])
    train_master = feature_matrix_full_history[feature_matrix_full_history["Weekly_Sales"].notna()]
    train_w, valid_w = split_by_date_ratio(train_master, ratio=2 / 3)
    print(f"train_window: {train_w.shape}, valid_window: {valid_w.shape}")

    # Macro của valid_window bị ghi đè bằng giá trị cách đây 52 tuần (mô phỏng
    # "không biết macro hiện tại" khi đánh giá) — CHỈ áp dụng cho valid_w,
    # train_w giữ nguyên macro thật đồng thời (bất đối xứng có chủ đích).
    valid_w = apply_macro_lag52_to_valid(valid_w, full_macro_history=feature_matrix)

    y_train = train_w["Weekly_Sales"].to_numpy()
    y_valid_true = valid_w["Weekly_Sales"].to_numpy()
    is_holiday_valid = valid_w["IsHoliday"].to_numpy()

    X_train = train_w.drop(columns=["Weekly_Sales"], errors="ignore")
    X_valid = valid_w.drop(columns=["Weekly_Sales"], errors="ignore")

    print("\n=== 4. Baseline Models ===")
    models = {
        "naive_same_week_last_year": NaiveSameWeekLastYear(),
        "simple_decision_tree": SimpleDecisionTreeBaseline(**baseline_cfg["baseline"]["decision_tree"]),
    }

    metrics_summary: dict[str, dict[str, float]] = {}
    history_rows: list[dict[str, object]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid).to_numpy()
        wmae = weighted_mae(y_valid_true, y_pred, is_holiday_valid)
        wmape = weighted_mape(y_valid_true, y_pred, is_holiday_valid)
        print(f"[{name}] WMAE={wmae:.2f}  WMAPE={wmape:.4f}")
        metrics_summary[name] = {"wmae": wmae, "wmape": wmape}
        history_rows.append({"model_name": name, "wmae": wmae, "wmape": wmape})

        if isinstance(model, SimpleDecisionTreeBaseline):
            fig_path = plot_decision_tree(
                model, run_ctx.reports_path("figures", "decision_tree.png")
            )
            print(f"  -> Da luu so do cay quyet dinh: {fig_path}")

    n_near_zero = int((np.abs(y_valid_true) < 1.0).sum())
    print(
        f"\n[LUU Y] {n_near_zero} dong trong valid_window co |Weekly_Sales| < 1.0 "
        "-> WMAPE bi anh huong manh boi cac dong nay (xem docstring "
        "weighted_mape trong evaluation/metrics.py). Dung WMAE lam metric on dinh chinh."
    )

    run_ctx.write_manifest(
        {
            "config_snapshot": {"data": data_cfg, "baseline": baseline_cfg},
            "metrics_summary": metrics_summary,
            "split_method": "ratio_2_3_of_unique_dates",
        }
    )
    run_ctx.append_run_history(history_rows)

    print(f"\n=== KET QUA: Baseline pipeline chay thanh cong. run_id={run_ctx.run_id} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
