"""Wrapper Direct multi-step (Giai đoạn 5) — khớp
notebooks/viet/multi_step/direct_way/direct_multimodel_DTree.ipynb và
direct_multimodel_rf.ipynb: fit N estimator độc lập, mỗi estimator dự báo
đúng 1 horizon h (h=1..horizon), dùng chung feature set cho mọi h.

NGOẠI LỆ có chủ đích so với interface chuẩn .fit(X,y)->self /
.predict(X)->pd.Series (CLAUDE.md mục 4 rule 4, A12): interface chuẩn giả
định 1 target duy nhất, không phù hợp N target đồng thời của Direct
multi-step. API multi-target dưới đây KHÔNG áp dụng cho model single-step
khác (NaiveSameWeekLastYear/SimpleDecisionTreeBaseline trong baseline.py giữ
nguyên interface gốc, không đổi).
"""

from typing import Callable, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class DirectMultiHorizonModel:
    """Quản lý N estimator độc lập theo horizon, dùng chung 1
    base_estimator_factory — tránh trùng lặp logic quản lý model con giữa
    Decision Tree và Random Forest (2 notebook chỉ khác estimator +
    hyperparameter, logic vòng lặp per-horizon giống hệt nhau)."""

    def __init__(
        self, base_estimator_factory: Callable[[], object], horizon: int = 10
    ) -> None:
        self._factory = base_estimator_factory
        self.horizon = horizon
        self._models: Dict[int, object] = {}
        self._feature_cols: Optional[List[str]] = None

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert category (Store) -> .cat.codes, giữ numeric, fillna(0.0) —
        pattern tương đương SimpleDecisionTreeBaseline._prepare_features
        (models/baseline.py) — COPY logic (không import chéo) để giữ 2 module
        độc lập, tránh coupling ẩn giữa baseline.py và file này."""
        df = X.copy()
        cat_cols = df.select_dtypes(include="category").columns
        for col in cat_cols:
            df[col] = df[col].cat.codes
        return df.select_dtypes(include="number").fillna(0.0)

    def fit(self, X: pd.DataFrame, y_multi: pd.DataFrame) -> "DirectMultiHorizonModel":
        """y_multi: DataFrame có cột target_t+1..target_t+{horizon} (từ
        add_horizon_targets). Mỗi horizon h fit 1 estimator riêng, CHỈ trên
        các dòng y_multi[f"target_t+{h}"] không NaN (dropna theo TỪNG h —
        khớp notebook: train_c = train_set.dropna(subset=[target])).

        Raise ValueError rõ ràng nếu 1 horizon không còn dòng train nào sau
        dropna (dữ liệu quá ngắn so với horizon) — thay vì để sklearn ném lỗi
        khó hiểu ở tầng dưới (đúng nguyên tắc raise lỗi tường minh thay vì
        âm thầm/khó hiểu, CLAUDE.md mục 4)."""
        numeric_X = self._prepare_features(X)
        self._feature_cols = list(numeric_X.columns)
        for h in range(1, self.horizon + 1):
            target_col = f"target_t+{h}"
            mask = y_multi[target_col].notna()
            if mask.sum() == 0:
                raise ValueError(
                    f"Horizon h={h}: không còn dòng train nào sau khi dropna "
                    f"target_t+{h} — dữ liệu train quá ngắn so với horizon="
                    f"{self.horizon}. Cần thêm dữ liệu lịch sử hoặc giảm horizon."
                )
            model = self._factory()
            model.fit(numeric_X.loc[mask], y_multi.loc[mask, target_col])
            self._models[h] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Trả DataFrame index=X.index, cột target_t+1..target_t+{horizon} —
        mỗi hàng dự đoán từ CÙNG 1 as_of_date cho mọi horizon (đúng ngữ nghĩa
        notebook: X_val cố định, mỗi model[h] predict trên chính X_val đó)."""
        if not self._models:
            raise RuntimeError("Model chưa được fit")
        numeric_X = self._prepare_features(X).reindex(
            columns=self._feature_cols, fill_value=0.0
        ).fillna(0.0)
        preds = {
            f"target_t+{h}": pd.Series(model.predict(numeric_X), index=X.index)
            for h, model in self._models.items()
        }
        return pd.DataFrame(preds, index=X.index)

    def predict_horizon(self, X: pd.DataFrame, h: int) -> pd.Series:
        """Tiện ích: predict CHỈ 1 horizon cụ thể, trả pd.Series — dùng khi
        Evaluation Layer lặp qua từng horizon riêng lẻ (tránh gọi predict()
        rồi bỏ (horizon-1) cột không dùng mỗi lần)."""
        if h not in self._models:
            raise KeyError(f"Chưa fit model cho horizon h={h}")
        numeric_X = self._prepare_features(X).reindex(
            columns=self._feature_cols, fill_value=0.0
        ).fillna(0.0)
        return pd.Series(self._models[h].predict(numeric_X), index=X.index)


def make_direct_decision_tree(
    horizon: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    random_state: int,
) -> DirectMultiHorizonModel:
    """Tham số đọc từ configs/model_direct_multistep.yaml ở pipeline
    orchestration — KHÔNG hard-code default nghiệp vụ ở đây."""
    return DirectMultiHorizonModel(
        base_estimator_factory=lambda: DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
        horizon=horizon,
    )


def make_direct_random_forest(
    horizon: int,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    n_jobs: int,
    random_state: int,
) -> DirectMultiHorizonModel:
    return DirectMultiHorizonModel(
        base_estimator_factory=lambda: RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        horizon=horizon,
    )
