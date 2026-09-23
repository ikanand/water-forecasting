import pytest

from tests.conftest import ROOT
from water_forecasting.config import ProjectConfig, Tables


@pytest.mark.parametrize("env,catalog", [("dev", "water_dev"), ("qa", "water_qa"), ("prod", "water_prod")])
def test_each_env_gets_its_own_catalog(env, catalog):
    cfg = ProjectConfig.from_yaml(ROOT / "project_config.yml", env=env)
    assert cfg.catalog == catalog
    assert cfg.table(Tables.FEATURES) == f"{catalog}.gold.demand_features"
    assert cfg.registered_model_name == f"{catalog}.ml.water_demand_forecast"


def test_unknown_env_rejected():
    with pytest.raises(ValueError):
        ProjectConfig.from_yaml(ROOT / "project_config.yml", env="staging")


def test_lower_envs_have_less_history():
    span = {e: ProjectConfig.from_yaml(ROOT / "project_config.yml", env=e).history_start for e in ("dev", "qa", "prod")}
    assert span["prod"] < span["qa"] < span["dev"]
