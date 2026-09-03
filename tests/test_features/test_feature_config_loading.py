"""Test đọc configs/features.yaml runtime — gap kỹ thuật đã fix cùng lúc với
việc đổi granularity (Store, Date). Trước đây enabled_blocks được truyền cứng
qua Python list, features.yaml chỉ mang tính minh họa, không được code nào
đọc thật — vi phạm CLAUDE.md mục 4 rule 3 "bật/tắt qua config, không hard-code
trong src/". Xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"."""

import yaml

from sales_forecast.features.pipeline import (
    ALL_BLOCKS,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    load_enabled_blocks_from_config,
    load_lag_rolling_params_from_config,
)


def test_load_enabled_blocks_reads_all_enabled_true(tmp_path):
    """Khi mọi block enabled=true trong YAML, hàm phải trả về đúng ALL_BLOCKS
    (không thiếu, không thừa, thứ tự không quan trọng)."""
    cfg_path = tmp_path / "features.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "feature_blocks": {
                "lag_rolling": {"enabled": True},
                "calendar": {"enabled": True},
                "markdown_promo": {"enabled": True},
                "store_encoding": {"enabled": True},
                "macro": {"enabled": True},
            }
        }),
        encoding="utf-8",
    )
    enabled = load_enabled_blocks_from_config(cfg_path)
    assert set(enabled) == set(ALL_BLOCKS)


def test_load_enabled_blocks_respects_disabled_block(tmp_path):
    """Block enabled=false trong YAML không được xuất hiện trong danh sách trả về."""
    cfg_path = tmp_path / "features.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "feature_blocks": {
                "lag_rolling": {"enabled": True},
                "calendar": {"enabled": True},
                "markdown_promo": {"enabled": False},
                "store_encoding": {"enabled": True},
                "macro": {"enabled": True},
            }
        }),
        encoding="utf-8",
    )
    enabled = load_enabled_blocks_from_config(cfg_path)
    assert "markdown" not in enabled
    assert "lag_rolling" in enabled


def test_load_enabled_blocks_from_real_config_file():
    """configs/features.yaml thật trong repo phải đọc được và trả về đúng
    ALL_BLOCKS (mọi block hiện đang enabled=true theo cấu hình mặc định)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    enabled = load_enabled_blocks_from_config(repo_root / "configs" / "features.yaml")
    assert set(enabled) == set(ALL_BLOCKS)


def test_load_lag_rolling_params_reads_from_yaml(tmp_path):
    """A51: đọc đúng lags/rolling_windows tùy chỉnh từ YAML, không dùng default
    khi key đã tồn tại."""
    cfg_path = tmp_path / "features.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "feature_blocks": {
                "lag_rolling": {"enabled": True, "lags": [1, 2], "rolling_windows": [3]},
            }
        }),
        encoding="utf-8",
    )
    lags, rolling_windows = load_lag_rolling_params_from_config(cfg_path)
    assert lags == [1, 2]
    assert rolling_windows == [3]


def test_load_lag_rolling_params_falls_back_to_defaults_when_missing(tmp_path):
    """A51: nếu YAML không có key lags/rolling_windows, fallback đúng về
    DEFAULT_LAGS/DEFAULT_ROLLING_WINDOWS (backward-compat)."""
    cfg_path = tmp_path / "features.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"feature_blocks": {"lag_rolling": {"enabled": True}}}),
        encoding="utf-8",
    )
    lags, rolling_windows = load_lag_rolling_params_from_config(cfg_path)
    assert lags == DEFAULT_LAGS
    assert rolling_windows == DEFAULT_ROLLING_WINDOWS


def test_load_lag_rolling_params_from_real_config_file():
    """A51: configs/features.yaml thật trong repo đọc được, khớp giá trị
    hiện tại đã khai báo trong YAML."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    lags, rolling_windows = load_lag_rolling_params_from_config(
        repo_root / "configs" / "features.yaml"
    )
    assert lags == [1, 4, 12, 26, 52]
    assert rolling_windows == [4, 12, 26]


def test_lag_rolling_params_match_previous_hardcoded_defaults():
    """A52 (regression test): đổi từ hard-code sang đọc YAML không được làm
    đổi kết quả feature_matrix khi giá trị YAML = giá trị default cũ."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    lags, rolling_windows = load_lag_rolling_params_from_config(
        repo_root / "configs" / "features.yaml"
    )
    assert lags == DEFAULT_LAGS
    assert rolling_windows == DEFAULT_ROLLING_WINDOWS
