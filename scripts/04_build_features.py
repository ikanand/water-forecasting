"""silver -> gold feature table. backfill = full rebuild, incremental = recent days + tomorrow."""

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import build_features
from water_forecasting.runtime import load_config, parse_args

args = parse_args([("--mode", {"choices": ["backfill", "incremental"], "default": "incremental"})])
cfg = load_config(args)
build_features(get_spark(), cfg, args.mode)
