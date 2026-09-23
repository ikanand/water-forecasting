"""Fail the run loudly (so alerts fire) when a gate blocked the pipeline."""

import argparse

p = argparse.ArgumentParser()
p.add_argument("--message", default="Pipeline blocked by a gate")
args, _ = p.parse_known_args()
raise SystemExit(f"BLOCKED: {args.message}. See the monitoring.dq_results table for details.")
