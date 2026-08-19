"""Ghép các feature block theo configs/features.yaml — Giai đoạn 3.

Mỗi block bật/tắt độc lập (CLAUDE.md mục 4 rule 3) — tắt 1 block không
được phá vỡ các block khác (A9, docs/05_test_plan.md mục 2.5).
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from sales_forecast.features.calendar import add_calendar_features
from sales_forecast.features.lag_rolling import add_lag_features, add_rolling_features
from sales_forecast.features.macro import add_macro_features
from sales_forecast.features.markdown_promo import add_markdown_features
from sales_forecast.features.store_encoding import encode_store

ALL_BLOCKS = ["lag_rolling", "calendar", "markdown", "encoding", "macro"]

# Mapping giữa tên block ngắn dùng nội bộ (build_feature_matrix) và tên key
# thật trong configs/features.yaml (feature_blocks.<key>.enabled) — 2 bộ tên
# lệch nhau do lịch sử đặt tên khác thời điểm, không đổi lại tên trong YAML
# để tránh phá vỡ tài liệu/spec đã tham chiếu tên đó.
_BLOCK_NAME_TO_YAML_KEY = {
    "lag_rolling": "lag_rolling",
    "calendar": "calendar",
    "markdown": "markdown_promo",
    "encoding": "store_encoding",
    "macro": "macro",
}

DEFAULT_LAGS = [1, 4, 12, 26, 52]
DEFAULT_ROLLING_WINDOWS = [4, 12, 26]


def load_enabled_blocks_from_config(config_path: str | Path) -> List[str]:
    """Đọc configs/features.yaml và trả về danh sách block đang enabled=true.

    Trước đây enabled_blocks được truyền cứng qua Python list ở nơi gọi
    (pipelines/, tests/), features.yaml chỉ mang tính minh họa, không được
    code nào đọc thật — vi phạm CLAUDE.md mục 4 rule 3 "bật/tắt qua config,
    không hard-code trong src/". Hàm này là nguồn enabled_blocks mặc định
    cho pipeline orchestration.
    """
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    blocks_cfg = raw_cfg.get("feature_blocks", {})
    enabled = []
    for block_name, yaml_key in _BLOCK_NAME_TO_YAML_KEY.items():
        block_entry = blocks_cfg.get(yaml_key, {})
        if block_entry.get("enabled", False):
            enabled.append(block_name)
    return enabled


def _add_has_history(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Flag has_history=True nếu Store xuất hiện trong train_df.

    Cold-start (docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"): ở
    granularity (Store, Date) hiện tại, đã xác nhận KHÔNG có Store nào
    cold-start ở test (cả 45 Store đều có trong train). Hàm này giữ lại để
    tổng quát hóa/phòng thủ (defensive) — nếu tương lai xuất hiện Store hoàn
    toàn mới, has_history vẫn phải được đánh dấu False đúng, không âm thầm
    dùng lag từ Store khác.
    """
    known_stores = train_df[["Store"]].drop_duplicates()
    known_stores = known_stores.assign(has_history=True)
    out = df.merge(known_stores, on=["Store"], how="left")
    out["has_history"] = out["has_history"].fillna(False)
    return out


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features_df: pd.DataFrame,
    enabled_blocks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ghép feature_matrix cho train_df + test_df (đã concat) theo các block bật.

    Lag/Rolling chỉ tính được cho phần có lịch sử Weekly_Sales — với test_df
    (không có Weekly_Sales), lag sẽ là NaN, phản ánh đúng nghĩa "chưa dự báo".
    """
    if enabled_blocks is None:
        enabled_blocks = ALL_BLOCKS

    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    combined = _add_has_history(combined, train_df)

    if "lag_rolling" in enabled_blocks:
        combined = add_lag_features(combined, group_cols=["Store"], lags=DEFAULT_LAGS)
        combined = add_rolling_features(
            combined, group_cols=["Store"], windows=DEFAULT_ROLLING_WINDOWS
        )

    if "calendar" in enabled_blocks:
        combined = add_calendar_features(combined)

    if "markdown" in enabled_blocks:
        combined = add_markdown_features(combined)

    if "macro" in enabled_blocks:
        combined = add_macro_features(combined, features_df)

    if "encoding" in enabled_blocks:
        combined = encode_store(combined)

    return combined
