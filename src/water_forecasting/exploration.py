"""Load the raw source tables for the exploration notebooks (not used by the pipeline).

On Databricks (a Spark session exists) it reads Unity Catalog directly. Locally it runs the same
query through a SQL warehouse with the Databricks SDK and caches the result as parquet, so a
notebook re-run needs neither a cluster nor network access.

Local auth follows the Databricks SDK defaults: set DATABRICKS_CONFIG_PROFILE (e.g. the profile
created by `databricks auth login`) and optionally DATABRICKS_WAREHOUSE_ID.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from water_forecasting.config import ProjectConfig

TIMESTAMP_COLS = {
    "consumption": ["timestamp_local", "timestamp_utc"],
    "weather_obs": ["timestamp_local", "timestamp_utc"],
    "weather_fcst": ["forecast_issued_local", "timestamp_local"],
    "calendar": ["date"],
}
_NUMERIC = {"DOUBLE", "FLOAT", "DECIMAL", "INT", "LONG", "SHORT", "BYTE"}


def source_tables(cfg: ProjectConfig) -> dict[str, str]:
    w = cfg.source.warehouse
    return {k: f"{w.catalog}.{w.schema_}.{getattr(w, f'{k}_table')}" for k in TIMESTAMP_COLS}


def _query_sql_warehouse(sql: str) -> pd.DataFrame:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    w = WorkspaceClient()
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID") or next(iter(w.warehouses.list())).id
    r = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
    )
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Query failed ({r.status.state}): {r.status.error}")

    columns = r.manifest.schema.columns
    rows, chunk = list(r.result.data_array or []), r.result
    while chunk.next_chunk_index is not None:
        chunk = w.statement_execution.get_statement_result_chunk_n(r.statement_id, chunk.next_chunk_index)
        rows += chunk.data_array or []

    df = pd.DataFrame(rows, columns=[c.name for c in columns])
    for c in columns:  # JSON_ARRAY returns every value as a string
        kind = c.type_name.value if c.type_name else ""
        if kind in _NUMERIC:
            df[c.name] = pd.to_numeric(df[c.name])
        elif kind == "BOOLEAN":
            df[c.name] = df[c.name].map({"true": True, "false": False})
    return df


def load_source(cfg: ProjectConfig, spark=None, cache_dir: str | Path | None = None, refresh: bool = False):
    """The four raw source tables (demand, observed weather, weather forecast, calendar), typed."""
    cache = Path(cache_dir) if cache_dir else None
    out = {}
    for key, table in source_tables(cfg).items():
        path = cache / f"{key}.parquet" if cache else None
        if path is not None and path.exists() and not refresh:
            df = pd.read_parquet(path)
        elif spark is not None:
            df = spark.table(table).toPandas()
        else:
            df = _query_sql_warehouse(f"SELECT * FROM {table}")
        for col in TIMESTAMP_COLS[key]:
            df[col] = pd.to_datetime(df[col])
        if path is not None and (refresh or not path.exists()):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        out[key] = df.sort_values(TIMESTAMP_COLS[key]).reset_index(drop=True)
    return out
