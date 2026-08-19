"""Test đọc configs/features.yaml runtime — gap kỹ thuật đã fix cùng lúc với
việc đổi granularity (Store, Date). Trước đây enabled_blocks được truyền cứng
qua Python list, features.yaml chỉ mang tính minh họa, không được code nào
đọc thật — vi phạm CLAUDE.md mục 4 rule 3 "bật/tắt qua config, không hard-code
trong src/". Xem docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"."""

import yaml

from sales_forecast.features.pipeline import ALL_BLOCKS, load_enabled_blocks_from_config


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
