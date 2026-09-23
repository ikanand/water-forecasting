"""Pull from the (mock) source systems into landing.

--mode backfill : load the configured history window (setup job; skipped if already loaded)
--mode daily    : 'receive' the next day of meter readings + weather, and tomorrow's weather forecast
--fault         : demo switch to break the data on purpose (none | missing_forecast | meter_outage)
"""

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import land_source
from water_forecasting.runtime import load_config, parse_args

args = parse_args(
    [("--mode", {"choices": ["backfill", "daily"], "default": "daily"}), ("--fault", {"default": "none"})]
)
cfg = load_config(args)
land_source(get_spark(), cfg, args.mode, args.fault)
