from __future__ import annotations

import os
from typing import Any


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _run_name(run: Any) -> str:
    return run.data.tags.get("mlflow.runName", "")


def runs_to_dataframe(
    runs: list[Any],
    limit: int = 200,
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    if metric_keys is None:
        metric_keys = ["loss", "val_loss", "accuracy"]

    rows: list[dict[str, Any]] = []
    for run in runs[:limit]:
        row: dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": _run_name(run),
            "status": run.info.status,
            "experiment_id": run.info.experiment_id,
            "start_time": run.info.start_time,
        }
        for key, value in run.data.params.items():
            row[f"param.{key}"] = value
        for key in metric_keys:
            row[key] = run.data.metrics.get(key)
        rows.append(row)
    return rows


def diagnose_run(run: Any, metric_key: str | None = None) -> dict[str, Any]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    selected_metric = metric_key
    if selected_metric is None:
        metric_names = sorted(run.data.metrics)
        selected_metric = metric_names[0] if metric_names else None

    history = client.get_metric_history(run.info.run_id, selected_metric) if selected_metric else []
    values = [item.value for item in history]
    best_point = min(history, key=lambda item: item.value) if history else None

    return {
        "run_id": run.info.run_id,
        "run_name": _run_name(run),
        "metric_key": selected_metric,
        "history_points": len(history),
        "latest_value": values[-1] if values else None,
        "best_value": best_point.value if best_point else None,
        "best_step": best_point.step if best_point else None,
    }


def compare_configs(run_a: Any, run_b: Any) -> list[dict[str, Any]]:
    params_a = run_a.data.params
    params_b = run_b.data.params
    all_keys = sorted(set(params_a) | set(params_b))
    diffs: list[dict[str, Any]] = []
    for key in all_keys:
        left = params_a.get(key)
        right = params_b.get(key)
        if left != right:
            diffs.append({"key": key, _run_name(run_a): left, _run_name(run_b): right})
    return diffs


def search_runs_by_experiment(
    experiment_name: str,
    filter_string: str = "",
    limit: int = 200,
    tracking_uri: str | None = None,
) -> list[Any]:
    import mlflow
    from mlflow import MlflowClient

    resolved_tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    return client.search_runs(
        [experiment.experiment_id],
        max_results=limit,
        filter_string=filter_string,
    )
