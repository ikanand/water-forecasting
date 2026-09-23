"""landing -> bronze (append-only, idempotent per batch, stamped with _ingested_at)."""

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import ingest_bronze
from water_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
ingest_bronze(get_spark(), cfg)
