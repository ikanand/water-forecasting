"""Unity Catalog I/O helpers. Everything Spark-specific lives here and in pipeline.py/training.py,
so the logic modules (synthetic, quality, features, modeling, decisions, monitoring) stay Spark-free."""

from __future__ import annotations

import logging

import pandas as pd

from water_forecasting.config import ProjectConfig, Tables
from water_forecasting.features import feature_columns

log = logging.getLogger(__name__)


def get_spark():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")  # all timestamps are naive UTC
    return spark


def exists(spark, table: str) -> bool:
    return spark.catalog.tableExists(table)


def read_pdf(spark, table: str, where: str | None = None) -> pd.DataFrame:
    df = spark.table(table)
    if where:
        df = df.where(where)
    return df.toPandas()


def _sdf(spark, pdf: pd.DataFrame):
    return spark.createDataFrame(pdf.reset_index(drop=True))


def append_pdf(spark, pdf: pd.DataFrame, table: str) -> None:
    if pdf.empty:
        return
    _sdf(spark, pdf).write.mode("append").option("mergeSchema", "true").saveAsTable(table)


def overwrite_pdf(spark, pdf: pd.DataFrame, table: str) -> None:
    _sdf(spark, pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)


def merge_pdf(spark, pdf: pd.DataFrame, table: str, keys: list[str]) -> None:
    """Idempotent upsert on natural keys - re-running a day never duplicates rows."""
    if pdf.empty:
        return
    if not exists(spark, table):
        _sdf(spark, pdf).write.saveAsTable(table)
        return
    view = "src_" + table.replace(".", "_")
    _sdf(spark, pdf).createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    spark.sql(
        f"MERGE INTO {table} t USING {view} s ON {on} WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *"
    )


# ------------------------------------------------------------------ setup ----
def create_schemas(spark, cfg: ProjectConfig) -> None:
    for key in cfg.schemas.model_dump():
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.schema(key)}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.schema('landing')}.{cfg.source.files_volume}")
    log.info("Schemas + volume ready in catalog %s", cfg.catalog)


def ensure_feature_table(spark, cfg: ProjectConfig) -> str:
    """A Delta table with a PRIMARY KEY (+ TIMESERIES) in UC *is* a feature table: it shows up in
    Catalog Explorer > Features, gets lineage, and supports point-in-time lookups."""
    name = cfg.table(Tables.FEATURES)
    cols = ",\n  ".join(f"{c} DOUBLE" for c in feature_columns(cfg.zones))
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {name} (
          zone_id STRING NOT NULL,
          target_ts TIMESTAMP NOT NULL,
          {cols},
          feature_computed_at TIMESTAMP,
          CONSTRAINT demand_features_pk PRIMARY KEY (zone_id, target_ts TIMESERIES)
        )
        COMMENT 'Day-ahead water demand features. Lags >= 48h; weather = forecast issued D-1 00:00.'
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    return name


def sim_today(spark, cfg: ProjectConfig) -> pd.Timestamp:
    """The simulated 'today': the day after the newest complete day of actuals in silver.
    In a real deployment this is simply today's date."""
    last = spark.sql(f"SELECT max(reading_ts) AS m FROM {cfg.table(Tables.SILVER_CONSUMPTION)}").first()["m"]
    if last is None:
        raise RuntimeError("Silver consumption is empty - run the setup job first.")
    return pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
