"""Baseline models for spatiotemporal biomass prediction."""

from baselines.models import QuantileRandomForest, QuantileXGBoost

__all__ = ["QuantileRandomForest", "QuantileXGBoost"]
