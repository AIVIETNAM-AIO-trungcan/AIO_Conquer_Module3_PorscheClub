"""Baseline models — Giai đoạn 4, cấu hình ở configs/model_baseline.yaml.

Cả 2 model tuân thủ cùng interface .fit(X, y)/.predict(X) để Evaluation
Layer dùng chung 1 đoạn code cho mọi model (CLAUDE.md mục 4 rule 4, A12).
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class NaiveSameWeekLastYear:
    """Naive baseline: dự báo = Weekly_Sales trung bình của cùng (Store,
    week_of_year) trong dữ liệu train. Fallback về trung bình toàn cục khi
    tổ hợp chưa từng xuất hiện (cold-start).

    Đơn vị dự báo (Store, week_of_year) — không còn Dept, xem
    docs/00_decisions.md [2026-08-19] "Đổi đơn vị dự báo"."""

    def __init__(self) -> None:
        self._lookup: pd.Series | None = None
        self._global_mean: float | None = None

    def fit(self, X: pd.DataFrame, y) -> "NaiveSameWeekLastYear":
        df = X.copy()
        df["_y"] = pd.Series(y).to_numpy()
        if "week_of_year" not in df.columns:
            df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        self._lookup = df.groupby(["Store", "week_of_year"])["_y"].mean()
        self._global_mean = float(df["_y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._lookup is None:
            raise RuntimeError("Model chưa được fit")
        df = X.copy()
        if "week_of_year" not in df.columns:
            df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        keys = pd.MultiIndex.from_frame(df[["Store", "week_of_year"]])
        preds = keys.map(self._lookup)
        return pd.Series(preds, index=X.index).fillna(self._global_mean)


class SimpleDecisionTreeBaseline:
    """Decision Tree đơn giản (docs/02_pipeline_architecture.md Giai đoạn 4),
    tham số từ configs/model_baseline.yaml."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ) -> None:
        self._model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._feature_cols: list[str] | None = None

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert cột category (Store/Dept) sang mã số trước khi lọc numeric —
        sklearn DecisionTreeRegressor không nhận dtype category trực tiếp."""
        df = X.copy()
        cat_cols = df.select_dtypes(include="category").columns
        for col in cat_cols:
            df[col] = df[col].cat.codes
        return df.select_dtypes(include="number").fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleDecisionTreeBaseline":
        numeric_X = self._prepare_features(X)
        self._feature_cols = list(numeric_X.columns)
        self._model.fit(numeric_X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._feature_cols is None:
            raise RuntimeError("Model chưa được fit")
        numeric_X = self._prepare_features(X).reindex(columns=self._feature_cols, fill_value=0.0).fillna(0.0)
        preds = self._model.predict(numeric_X)
        return pd.Series(preds, index=X.index)
