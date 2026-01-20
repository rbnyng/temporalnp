"""
Baseline ML models for biomass prediction.

Provides Quantile Random Forest and Quantile XGBoost implementations
for uncertainty-aware predictions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QUANTILE_FOREST = True
except ImportError:
    HAS_QUANTILE_FOREST = False


class QuantileRandomForest:
    """
    Quantile Regression Forest wrapper using the quantile-forest package.

    Uses the proper QRF implementation from Meinshausen (2006) which stores
    all leaf values during training for exact conditional quantile estimation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
        default_quantiles: Tuple[float, float] = (0.025, 0.975)
    ):
        if not HAS_QUANTILE_FOREST:
            raise ImportError(
                "quantile-forest package not installed. "
                "Install with: pip install quantile-forest"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.default_quantiles = default_quantiles

        self.model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the quantile regression forest."""
        y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return median prediction."""
        return self.model.predict(X, quantiles=0.5)

    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with standard deviation estimate from quantile range.

        Returns:
            (predictions, std) where std is estimated from quantile range
        """
        # Predict median as point estimate
        predictions = self.model.predict(X, quantiles=0.5)

        # Use default quantiles to estimate std
        lower_q, upper_q = self.default_quantiles
        quantile_preds = self.model.predict(X, quantiles=[lower_q, upper_q])
        lower = quantile_preds[:, 0]
        upper = quantile_preds[:, 1]

        # Approximate std from quantile range using normal distribution z-scores
        z_upper = norm.ppf(upper_q)
        z_lower = norm.ppf(lower_q)
        std = (upper - lower) / (z_upper - z_lower)

        return predictions, std

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float]
    ) -> np.ndarray:
        """
        Predict specific quantiles.

        Args:
            X: Features (n_samples, n_features)
            quantiles: List of quantiles to predict

        Returns:
            (n_samples, n_quantiles) array of predictions
        """
        return self.model.predict(X, quantiles=quantiles)


class QuantileXGBoost:
    """
    Quantile XGBoost implementation using native quantile regression.

    Trains separate models for each quantile using XGBoost's built-in
    quantile loss function (reg:quantileerror).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
        quantiles: Tuple[float, ...] = (0.159, 0.5, 0.841)
    ):
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.quantiles = quantiles

        self.models: Dict[float, xgb.XGBRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit a separate model for each quantile."""
        y = np.asarray(y).ravel()

        for q in tqdm(self.quantiles, desc="Training quantile models"):
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                objective='reg:quantileerror',
                quantile_alpha=q,
                verbosity=0
            )
            model.fit(X, y)
            self.models[q] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return median prediction."""
        return self.models[0.5].predict(X)

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Predict quantiles for each sample.

        Args:
            X: Features (n_samples, n_features)
            quantiles: List of quantiles to predict (must be subset of trained quantiles)

        Returns:
            Predictions of shape (n_samples, n_quantiles)
        """
        if quantiles is None:
            quantiles = list(self.quantiles)

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(quantiles)))

        for i, q in enumerate(quantiles):
            if q not in self.models:
                raise ValueError(
                    f"Quantile {q} not trained. Available: {list(self.models.keys())}"
                )
            predictions[:, i] = self.models[q].predict(X)

        return predictions
