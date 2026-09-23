"""MLflow model wrapper + Unity Catalog registry helpers (aliases: @champion, @challenger)."""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from water_forecasting.config import ProjectConfig

log = logging.getLogger(__name__)


class WaterDemandModel(mlflow.pyfunc.PythonModel):
    """Selects the right columns, predicts, and never returns negative demand."""

    def __init__(self, estimator, feature_columns: list[str], family: str):
        self.estimator = estimator
        self.feature_columns = feature_columns
        self.family = family

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        X = model_input[self.feature_columns].astype(float)
        return np.clip(self.estimator.predict(X), 0, None)


def use_unity_catalog() -> MlflowClient:
    mlflow.set_registry_uri("databricks-uc")
    return MlflowClient()


def get_alias_version(cfg: ProjectConfig, alias: str):
    client = use_unity_catalog()
    try:
        return client.get_model_version_by_alias(cfg.registered_model_name, alias)
    except Exception:  # alias or model does not exist yet
        return None


def load_alias(cfg: ProjectConfig, alias: str):
    mv = get_alias_version(cfg, alias)
    if mv is None:
        return None, None
    return mlflow.pyfunc.load_model(f"models:/{cfg.registered_model_name}@{alias}"), mv


def set_alias(cfg: ProjectConfig, alias: str, version: str) -> None:
    use_unity_catalog().set_registered_model_alias(cfg.registered_model_name, alias, version)
    log.info("%s@%s -> v%s", cfg.registered_model_name, alias, version)
