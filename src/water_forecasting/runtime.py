"""Plumbing shared by the thin scripts in scripts/: args, config, logging, task values."""

from __future__ import annotations

import argparse
import logging

from water_forecasting.config import ProjectConfig


def parse_args(extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", required=True, help="${workspace.root_path} of the deployed bundle")
    p.add_argument("--env", required=True, choices=["dev", "qa", "prod"])
    p.add_argument("--git_sha", default="local")
    p.add_argument("--job_run_id", default="manual")
    for name, kwargs in extra or []:
        p.add_argument(name, **kwargs)
    args, _unknown = p.parse_known_args()
    return args


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s", force=True)
    logging.getLogger("py4j").setLevel(logging.WARNING)


def load_config(args: argparse.Namespace) -> ProjectConfig:
    setup_logging()
    cfg = ProjectConfig.from_yaml(f"{args.root_path}/files/project_config.yml", env=args.env)
    logging.getLogger(__name__).info("env=%s catalog=%s", cfg.env, cfg.catalog)
    return cfg


def set_task_value(key: str, value) -> None:
    """Expose a value to downstream tasks (read by condition_task via {{tasks.<task>.values.<key>}})."""
    from pyspark.dbutils import DBUtils

    from water_forecasting.lakehouse import get_spark

    DBUtils(get_spark()).jobs.taskValues.set(key=key, value=value)
    logging.getLogger(__name__).info("task value %s=%s", key, value)
