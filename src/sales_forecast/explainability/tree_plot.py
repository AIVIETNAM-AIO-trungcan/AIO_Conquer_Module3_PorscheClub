"""Vẽ cây quyết định — hỗ trợ hiểu sơ bộ luồng quyết định của
SimpleDecisionTreeBaseline (docs/02_pipeline_architecture.md Giai đoạn 4).

Đây là trực quan hóa mô tả cấu trúc cây đã fit, KHÔNG phải TreeSHAP
(shap_runner.py) — dùng để đọc nhanh các ngưỡng split, không thay thế
phân tích SHAP ở Giai đoạn 8.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sales_forecast.models.baseline import SimpleDecisionTreeBaseline


def plot_decision_tree(
    model: SimpleDecisionTreeBaseline,
    output_path: str | Path,
    max_depth_display: int | None = None,
) -> Path:
    """Vẽ cây quyết định đã fit và lưu ra file ảnh.

    model: SimpleDecisionTreeBaseline đã gọi .fit(...).
    output_path: đường dẫn file ảnh đích (vd. reports/figures/decision_tree.png).
    max_depth_display: giới hạn số tầng hiển thị (None = hiển thị toàn bộ cây đã train).
    """
    if model._feature_cols is None:
        raise RuntimeError("Model chưa được fit — gọi .fit(X, y) trước khi vẽ")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model._model,
        feature_names=model._feature_cols,
        filled=True,
        rounded=True,
        max_depth=max_depth_display,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
