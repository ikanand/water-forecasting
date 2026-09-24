"""Guards against the bundle (databricks.yml) and the runtime config (project_config.yml) drifting apart."""

import json

import yaml

from tests.conftest import ROOT
from water_forecasting.config import ProjectConfig

BUNDLE = yaml.safe_load((ROOT / "databricks.yml").read_text())


def test_bundle_catalog_matches_project_config():
    default = BUNDLE["variables"]["catalog"]["default"]
    for target, spec in BUNDLE["targets"].items():
        bundle_catalog = spec.get("variables", {}).get("catalog", default)
        assert bundle_catalog == ProjectConfig.from_yaml(ROOT / "project_config.yml", env=target).catalog, target


def test_qa_and_prod_run_as_service_principals_from_main_only():
    for target in ("qa", "prod"):
        spec = BUNDLE["targets"][target]
        assert spec["mode"] == "production"
        assert spec["run_as"]["service_principal_name"]
        assert spec["git"]["branch"] == "main"
    assert BUNDLE["targets"]["qa"]["run_as"] != BUNDLE["targets"]["prod"]["run_as"]


def test_only_prod_schedules_are_live():
    assert BUNDLE["variables"]["schedule_pause_status"]["default"] == "PAUSED"
    assert BUNDLE["targets"]["prod"]["variables"]["schedule_pause_status"] == "UNPAUSED"
    assert "schedule_pause_status" not in BUNDLE["targets"]["qa"].get("variables", {})


def test_dashboard_queries_use_catalog_free_names():
    """dataset_catalog sets the catalog per target, so queries must not hard-code one."""
    dash = json.loads((ROOT / "resources" / "dashboards" / "water_forecasting.lvdash.json").read_text())
    sql = " ".join(line for ds in dash["datasets"] for line in ds["queryLines"])
    assert "ewec_demo_" not in sql
    widgets = [w["widget"] for p in dash["pages"] for w in p["layout"]]
    names = {ds["name"] for ds in dash["datasets"]}
    for w in widgets:
        for q in w.get("queries", []):
            assert q["query"]["datasetName"] in names
