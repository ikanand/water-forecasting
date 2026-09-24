"""Typed project configuration loaded from project_config.yml for one environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ENVS = ("dev", "qa", "prod")


class Schemas(BaseModel):
    landing: str = "landing"
    bronze: str = "bronze"
    silver: str = "silver"
    gold: str = "gold"
    ml: str = "ml"
    monitoring: str = "monitoring"


class WarehouseSourceConfig(BaseModel):
    """mode=warehouse: another Unity Catalog catalog treated as the raw source system."""

    catalog: str = "mlops_dev"
    schema_: str = Field("bronze", alias="schema")
    consumption_table: str = "dubai_water_demand_hourly_raw"
    weather_obs_table: str = "dubai_weather_hourly_raw"
    weather_fcst_table: str = "dubai_weather_forecast_hourly_raw"
    calendar_table: str = "uae_calendar_events_daily_raw"

    model_config = {"populate_by_name": True}


class SourceConfig(BaseModel):
    mode: Literal["warehouse", "files"] = "warehouse"
    files_volume: str = "raw_files"
    warehouse: WarehouseSourceConfig = Field(default_factory=WarehouseSourceConfig)


class ForecastConfig(BaseModel):
    timezone: str = "Asia/Dubai"  # local time of the source data and of "today"
    issue_hour_utc: int = 5
    horizon_hours: int = 24
    min_lag_hours: int = 48


class ModelConfig(BaseModel):
    registered_model: str = "water_demand_forecast"
    candidates: list[str] = Field(default_factory=lambda: ["seasonal_naive_168", "ridge_fourier", "lightgbm"])
    holdout_days: int = 28
    ridge_alpha: float = 1.0
    lgbm_params: dict[str, Any] = Field(default_factory=dict)


class DecisionConfig(BaseModel):
    min_improvement_pct: float = 1.0
    max_acceptable_mape: float = 15.0
    degradation_factor: float = 1.25
    degradation_days: int = 3
    min_fresh_days: int = 7


class QualityConfig(BaseModel):
    demand_min_m3h: float = 0.0
    demand_max_m3h: float = 200000.0
    temperature_min_c: float = 0.0
    temperature_max_c: float = 50.0
    max_missing_hours_pct: float = 2.0
    frozen_min_hours: int = 3


class Tables:
    """Logical table names -> (schema key, table name)."""

    # landing = raw pull from the source catalog, as-received
    LANDING_CONSUMPTION = ("landing", "water_demand_hourly")
    LANDING_WEATHER_OBS = ("landing", "weather_hourly")
    LANDING_WEATHER_FCST = ("landing", "weather_forecast_hourly")
    LANDING_CALENDAR = ("landing", "calendar_daily")
    # bronze = raw + lineage columns
    BRONZE_CONSUMPTION = ("bronze", "water_demand_hourly")
    BRONZE_WEATHER_OBS = ("bronze", "weather_hourly")
    BRONZE_WEATHER_FCST = ("bronze", "weather_forecast_hourly")
    BRONZE_CALENDAR = ("bronze", "calendar_daily")
    # silver = validated
    SILVER_CONSUMPTION = ("silver", "demand_hourly")
    SILVER_WEATHER_OBS = ("silver", "weather_hourly")
    SILVER_WEATHER_FCST = ("silver", "weather_forecast_hourly")
    SILVER_CALENDAR = ("silver", "calendar_daily")
    # gold
    FEATURES = ("gold", "demand_features")
    FORECASTS = ("gold", "demand_forecasts")
    # monitoring
    DQ_RESULTS = ("monitoring", "dq_results")
    QUARANTINE = ("monitoring", "dq_quarantine")
    FORECAST_ACCURACY = ("monitoring", "forecast_accuracy_hourly")
    DAILY_ACCURACY = ("monitoring", "forecast_accuracy_daily")
    MODEL_SELECTION = ("monitoring", "model_selection")


# bronze table -> (landing table, silver table, natural key)
BRONZE_FLOW = {
    "consumption": (Tables.LANDING_CONSUMPTION, Tables.BRONZE_CONSUMPTION),
    "weather_obs": (Tables.LANDING_WEATHER_OBS, Tables.BRONZE_WEATHER_OBS),
    "weather_fcst": (Tables.LANDING_WEATHER_FCST, Tables.BRONZE_WEATHER_FCST),
    "calendar": (Tables.LANDING_CALENDAR, Tables.BRONZE_CALENDAR),
}


class ProjectConfig(BaseModel):
    env: Literal["dev", "qa", "prod"]
    project_name: str
    catalog: str
    history_start: str
    history_end: str
    schemas: Schemas = Field(default_factory=Schemas)
    source: SourceConfig = Field(default_factory=SourceConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    decisions: DecisionConfig = Field(default_factory=DecisionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    refresh_source_catalog: str = "ewec_demo_prod"
    refresh_lookback_days: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, env: str) -> ProjectConfig:
        if env not in ENVS:
            raise ValueError(f"env must be one of {ENVS}, got {env!r}")
        raw = yaml.safe_load(Path(path).read_text())
        env_block = raw.pop("environments")[env]
        refresh = raw.pop("refresh", {}) or {}
        return cls(
            env=env,
            **raw,
            **env_block,
            refresh_source_catalog=refresh.get("source_catalog", "ewec_demo_prod"),
            refresh_lookback_days=(refresh.get("lookback_days") or {}).get(env),
        )

    # ---- naming helpers ----------------------------------------------------------
    def schema(self, key: str) -> str:
        return f"{self.catalog}.{getattr(self.schemas, key)}"

    def table(self, ref: tuple[str, str], catalog: str | None = None) -> str:
        schema_key, name = ref
        return f"{catalog or self.catalog}.{getattr(self.schemas, schema_key)}.{name}"

    @property
    def registered_model_name(self) -> str:
        return f"{self.catalog}.{self.schemas.ml}.{self.model.registered_model}"

    @property
    def files_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schemas.landing}/{self.source.files_volume}"

    def experiment_path(self, bundle_root: str) -> str:
        """MLflow experiment inside the deployed bundle folder (which always exists)."""
        return f"{bundle_root.removeprefix('/Workspace')}/{self.project_name}_{self.env}_experiment"
