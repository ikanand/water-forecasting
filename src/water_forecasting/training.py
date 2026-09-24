"""Weekly / triggered training: fit every candidate family, compare on a time-based holdout,
log everything to MLflow, register the winner as @challenger, promote to @champion only if it
beats the current champion on the SAME holdout."""

from __future__ import annotations

import logging

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from water_forecasting import features as F
from water_forecasting.config import ProjectConfig, Tables
from water_forecasting.decisions import fair_comparison_window, should_promote
from water_forecasting.lakehouse import append_pdf
from water_forecasting.modeling import evaluate, make_model, time_split
from water_forecasting.registry import WaterDemandModel, load_alias, set_alias, use_unity_catalog

log = logging.getLogger(__name__)


def load_training_frame(spark, cfg: ProjectConfig) -> pd.DataFrame:
    feats = spark.table(cfg.table(Tables.FEATURES)).alias("f")
    labels = spark.table(cfg.table(Tables.SILVER_CONSUMPTION)).selectExpr("timestamp_local AS target_ts", "demand_m3h")
    df = feats.join(labels, on=F.KEYS, how="inner").where("lag_504h IS NOT NULL").toPandas()
    return df.sort_values(F.KEYS).reset_index(drop=True)


def train(spark, cfg: ProjectConfig, experiment_path: str, trigger: str, git_sha: str, run_id: str) -> dict:
    use_unity_catalog()
    mlflow.set_experiment(experiment_path)
    cols = F.model_columns()
    data = load_training_frame(spark, cfg)
    tr, ho = time_split(data, cfg.model.holdout_days)
    log.info("Training rows %s, holdout rows %s (%s..%s)", len(tr), len(ho), ho.target_ts.min(), ho.target_ts.max())

    leaderboard, fitted = [], {}
    with mlflow.start_run(run_name=f"{cfg.env}-{trigger}-{pd.Timestamp.now():%Y%m%d-%H%M}") as parent:
        mlflow.set_tags({"env": cfg.env, "trigger": trigger, "git_sha": git_sha, "job_run_id": run_id})
        mlflow.log_params(
            {
                "train_start": str(tr.target_ts.min()),
                "train_end": str(tr.target_ts.max()),
                "holdout_start": str(ho.target_ts.min()),
                "holdout_end": str(ho.target_ts.max()),
                "n_train": len(tr),
                "n_holdout": len(ho),
                "n_features": len(cols),
            }
        )

        for family in cfg.model.candidates:
            with mlflow.start_run(run_name=family, nested=True) as child:
                est = make_model(family, cfg.model.ridge_alpha, cfg.model.lgbm_params).fit(tr[cols], tr.demand_m3h)
                fitted[family] = est  # holdout-unseen version, used for the champion comparison
                pred = est.predict(ho[cols])
                metrics = evaluate(ho.demand_m3h, pred)
                mlflow.log_params({"family": family, **{f"p_{k}": v for k, v in est.get_params().items()}})
                if hasattr(est, "annual_growth_pct"):
                    metrics["fitted_annual_growth_pct"] = est.annual_growth_pct
                mlflow.log_metrics(metrics)
                imp = est.feature_importance(cols) if hasattr(est, "feature_importance") else None
                if imp is not None:
                    mlflow.log_dict(imp.head(20).round(1).to_dict(), "feature_importance_top20.json")
                leaderboard.append({"family": family, "mlflow_run_id": child.info.run_id, **metrics})
                log.info("%-20s MAPE %.2f%%  peak MAPE %.2f%%", family, metrics["mape"], metrics["peak_mape"])

        board = pd.DataFrame(leaderboard).sort_values("mape").reset_index(drop=True)
        best = board.iloc[0]
        mlflow.log_dict(board.round(3).to_dict(orient="records"), "leaderboard.json")

        # Refit the winning family on ALL data (incl. holdout) - that is the model we deploy.
        final_est = make_model(best.family, cfg.model.ridge_alpha, cfg.model.lgbm_params).fit(
            data[cols], data.demand_m3h
        )
        wrapped = WaterDemandModel(final_est, cols, best.family)
        example = data[cols].head(50)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped,
            signature=infer_signature(example, wrapped.predict(None, example)),
            input_example=example.head(3),
        )
        mv = mlflow.register_model(info.model_uri, cfg.registered_model_name)
        client = use_unity_catalog()
        for k, v in {
            "family": best.family,
            "holdout_mape": f"{best.mape:.4f}",
            "trigger": trigger,
            "trained_through": str(data.target_ts.max()),
            "git_sha": git_sha,
        }.items():
            client.set_model_version_tag(cfg.registered_model_name, mv.version, k, v)
        set_alias(cfg, "challenger", mv.version)

        # Champion vs challenger on holdout hours NEITHER has seen in training.
        champ_model, champ_mv = load_alias(cfg, "champion")
        champ_mape, chall_mape = None, float(best.mape)
        if champ_model is None:
            promote, reason = should_promote(
                chall_mape, None, cfg.decisions.min_improvement_pct, cfg.decisions.max_acceptable_mape
            )
        else:
            mask, window = fair_comparison_window(
                ho.target_ts, champ_mv.tags.get("trained_through"), cfg.decisions.min_fresh_days
            )
            if mask is None:
                promote, reason = False, window
            else:
                fresh = ho[mask.to_numpy()]
                champ_mape = evaluate(fresh.demand_m3h, champ_model.predict(fresh[cols]))["mape"]
                chall_mape = evaluate(fresh.demand_m3h, fitted[best.family].predict(fresh[cols]))["mape"]
                promote, reason = should_promote(
                    chall_mape, champ_mape, cfg.decisions.min_improvement_pct, cfg.decisions.max_acceptable_mape
                )
                reason = f"{reason}; {window}"
        if promote:
            set_alias(cfg, "champion", mv.version)
        mlflow.set_tags({"selected_family": best.family, "promoted": str(promote), "decision": reason})
        mlflow.log_metrics(
            {"challenger_mape": chall_mape, **({"champion_mape": champ_mape} if champ_mape is not None else {})}
        )
        log.info("Decision: promote=%s (%s)", promote, reason)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    board = board.assign(
        trained_at=now,
        env=cfg.env,
        trigger=trigger,
        job_run_id=run_id,
        parent_run_id=parent.info.run_id,
        selected=board.family == best.family,
        registered_version=str(mv.version),
        champion_version_before=str(champ_mv.version) if champ_mv else "none",
        champion_mape_on_holdout=float("nan") if champ_mape is None else champ_mape,
        promoted=promote & (board.family == best.family),
        decision=reason,
    )
    append_pdf(spark, board, cfg.table(Tables.MODEL_SELECTION))
    return {"version": mv.version, "family": best.family, "promoted": promote, "reason": reason}
