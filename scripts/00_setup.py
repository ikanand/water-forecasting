"""Create schemas + volume in this environment's catalog (idempotent)."""

from water_forecasting.lakehouse import create_schemas, get_spark
from water_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
create_schemas(get_spark(), cfg)
