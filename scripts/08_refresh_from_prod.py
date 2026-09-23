"""Quarterly: copy a recent contiguous window of prod silver into dev/qa (no-op in prod)."""

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import refresh_from_prod
from water_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
refresh_from_prod(get_spark(), cfg)
