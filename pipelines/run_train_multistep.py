"""Orchestration: load raw -> validate -> aggregate -> feature matrix ->
split (ty le 2/3) -> horizon targets -> fit Direct multi-step models
(Decision Tree + Random Forest, HORIZON=10) -> danh gia theo tung horizon.

Chi orchestration (CLAUDE.md muc 4 rule: pipelines/ khong chua logic nghiep
vu, moi logic that nam trong src/sales_forecast/).

Khop notebooks/viet/multi_step/direct_way/direct_multimodel_DTree.ipynb va
direct_multimodel_rf.ipynb - xem docs/00_decisions.md [2026-08-31]
"Direct multi-step HORIZON=10, khong chia nhom". Dung chung nen du lieu da
dong bo voi pipelines/run_train_baseline.py (Feature Engineering truoc
Split cuc bo, split ty le 2/3) - KHONG sua run_train_baseline.py.

Cach dung:
    python pipelines/run_train_multistep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sales_forecast.evaluation.metrics import wape, weighted_mae, weighted_mape
from sales_forecast.features.horizon_target import add_horizon_targets
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
from sales_forecast.models.direct_multihorizon import (
    make_direct_decision_tree,
    make_direct_random_forest,
)
from sales_forecast.splitting.ratio_split import split_by_date_ratio
from sales_forecast.utils.run_tracking import start_run


def main() -> int:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    multistep_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "model_direct_multistep.yaml").read_text(encoding="utf-8")
    )

    run_ctx = start_run(
        reports_base=REPO_ROOT / data_cfg["data_paths"]["reports_dir"],
        predictions_base=REPO_ROOT / data_cfg["data_paths"]["predictions_dir"],
        pipeline_name="train_multistep",
    )
    print(f"=== run_id: {run_ctx.run_id} ===")
    try:
        exit_code = _run(run_ctx, data_cfg, multistep_cfg)
    except BaseException:
        run_ctx.finalize(status="failed")
        raise
    run_ctx.finalize(status="success" if exit_code == 0 else "failed")
    return exit_code


def _select_feature_cols(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    """Loai Date, Weekly_Sales, va moi cot target_t+* khoi feature set -
    khong copy cung danh sach cot tu notebook cu, dung tren cot that cua
    feature_matrix hien tai."""
    drop_cols = {"Date", "Weekly_Sales"} | set(target_cols)
    return [c for c in df.columns if c not in drop_cols]


def _run(run_ctx, data_cfg: dict, multistep_cfg: dict) -> int:
    horizon = int(multistep_cfg["direct_multistep"]["horizon"])

    print("=== 1. Load & validate raw data ===")
    raw = load_raw_data(REPO_ROOT / data_cfg["data_paths"]["raw_dir"])
    validate_train_schema(raw["train"])
    validate_test_schema(raw["test"])
    validate_features_schema(raw["features"])
    validate_stores_schema(raw["stores"])
    print(f"train: {raw['train'].shape}, features: {raw['features'].shape}")

    print("\n=== 1b. Aggregate (Store, Dept, Date) -> (Store, Date) ===")
    train_agg, test_agg = aggregate_to_store_date(raw["train"], raw["test"])
    validate_train_aggregated_schema(train_agg)
    validate_test_aggregated_schema(test_agg)
    print(f"train (aggregated): {train_agg.shape}, test (aggregated): {test_agg.shape}")

    print("\n=== 2. Feature Engineering (chay TRUOC split - cung nen voi run_train_baseline.py) ===")
    enabled_blocks = load_enabled_blocks_from_config(REPO_ROOT / "configs" / "features.yaml")
    lags, rolling_windows = load_lag_rolling_params_from_config(REPO_ROOT / "configs" / "features.yaml")
    print(f"enabled_blocks (tu configs/features.yaml): {enabled_blocks}")
    feature_matrix = build_feature_matrix(
        train_agg, test_agg, raw["features"],
        enabled_blocks=enabled_blocks, lags=lags, rolling_windows=rolling_windows,
    )
    print(f"feature_matrix (truoc dropna lag_52w): {feature_matrix.shape}")

    print("\n=== 3. Split (theo ty le 2/3 so ngay duy nhat) ===")
    feature_matrix_full_history = feature_matrix.dropna(subset=["lag_52w"])
    train_master = feature_matrix_full_history[feature_matrix_full_history["Weekly_Sales"].notna()]
    train_w, valid_w = split_by_date_ratio(train_master, ratio=2 / 3)
    print(f"train_window: {train_w.shape}, valid_window: {valid_w.shape}")

    print(f"\n=== 4. Horizon Targets (HORIZON={horizon}, khop direct_multimodel_*.ipynb) ===")
    # Sinh target_t+1..target_t+{horizon} rieng cho train_w va valid_w -
    # KHONG gop chung roi shift (moi tap co bien Store/Date rieng, shift theo
    # dung group tranh leak cheo giua 2 tap).
    train_w_targets = add_horizon_targets(train_w, horizon=horizon)
    valid_w_targets = add_horizon_targets(valid_w, horizon=horizon)
    target_cols = [f"target_t+{h}" for h in range(1, horizon + 1)]

    feature_cols = _select_feature_cols(train_w_targets, target_cols)
    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")

    X_train = train_w_targets[feature_cols]
    y_train_multi = train_w_targets[target_cols]
    X_valid = valid_w_targets[feature_cols]
    is_holiday_valid = valid_w_targets["IsHoliday"].to_numpy()

    print("\n=== 5. Fit Direct multi-step models ===")
    dt_cfg = multistep_cfg["direct_multistep"]["decision_tree"]
    rf_cfg = multistep_cfg["direct_multistep"]["random_forest"]
    models = {
        "direct_decision_tree": make_direct_decision_tree(horizon=horizon, **dt_cfg),
        "direct_random_forest": make_direct_random_forest(horizon=horizon, **rf_cfg),
    }

    metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for model_name, model in models.items():
        model.fit(X_train, y_train_multi)
        model_wmae, model_wmape = [], []
        for h in range(1, horizon + 1):
            target_col = f"target_t+{h}"
            valid_mask = valid_w_targets[target_col].notna()
            if valid_mask.sum() == 0:
                continue
            y_true_h = valid_w_targets.loc[valid_mask, target_col].to_numpy()
            y_pred_h = model.predict_horizon(X_valid.loc[valid_mask], h).to_numpy()
            is_holiday_h = is_holiday_valid[valid_mask.to_numpy()]

            mae = mean_absolute_error(y_true_h, y_pred_h)
            rmse = root_mean_squared_error(y_true_h, y_pred_h)
            wape_h = wape(y_true_h, y_pred_h)
            wmae_h = weighted_mae(y_true_h, y_pred_h, is_holiday_h)
            wmape_h = weighted_mape(y_true_h, y_pred_h, is_holiday_h)
            model_wmae.append(wmae_h)
            model_wmape.append(wmape_h)

            metric_rows.append({
                "model": model_name, "horizon": h, "n_valid": int(valid_mask.sum()),
                "mae": mae, "rmse": rmse, "wape": wape_h,
                "wmae": wmae_h, "wmape": wmape_h,
            })
            print(f"[{model_name}] h={h:2d}  MAE={mae:.2f}  RMSE={rmse:.2f}  "
                  f"WAPE={wape_h:.4f}  WMAE={wmae_h:.2f}  WMAPE={wmape_h:.4f}")

        history_rows.append({
            "model_name": model_name,
            "wmae": sum(model_wmae) / len(model_wmae) if model_wmae else float("nan"),
            "wmape": sum(model_wmape) / len(model_wmape) if model_wmape else float("nan"),
        })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = run_ctx.reports_path("metrics", "multistep_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n-> Da luu ket qua theo horizon: {metrics_path}")

    run_ctx.write_manifest({
        "config_snapshot": {"data": data_cfg, "model_direct_multistep": multistep_cfg},
        "horizon": horizon,
        "models": list(models.keys()),
    })
    run_ctx.append_run_history(history_rows)

    print(f"\n=== KET QUA: Direct multi-step pipeline chay thanh cong. run_id={run_ctx.run_id} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
