from pathlib import Path

import pandas as pd
import pytest

from water_forecasting import synthetic
from water_forecasting.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]
ZONES = ["north", "central", "south"]


@pytest.fixture(scope="session")
def cfg() -> ProjectConfig:
    return ProjectConfig.from_yaml(ROOT / "project_config.yml", env="dev")


@pytest.fixture(scope="session")
def raw() -> dict[str, pd.DataFrame]:
    """~10 weeks of clean synthetic data."""
    return synthetic.generate_range("2025-03-01", "2025-05-10", ZONES, seed=7, backfill=True)


@pytest.fixture
def bronze(raw):
    return {k: v.assign(_ingested_at=pd.Timestamp("2026-01-01")) for k, v in raw.items()}
