<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# MLflow SDK Reference

Use the MLflow fluent API inside training code when you are logging params, metrics, tags, or artifacts. Use `MlflowClient` when you need low-level querying of runs or metric history.

## Canonical Usage

```python
import os

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("my-competition")

with mlflow.start_run(run_name="exp001"):
    mlflow.log_params({"lr": 0.01, "epochs": 10})
    mlflow.log_metric("cv_score", 0.8421)
    mlflow.log_artifacts("models/exp001/artifacts")

client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
runs = client.search_runs(["1"], filter_string="", max_results=50)
history = client.get_metric_history(runs[0].info.run_id, "cv_score")
```

## Fluent API vs `MlflowClient`

- Use the fluent API for run-scoped logging in `train.py`
- Use `MlflowClient` for searching runs, fetching metric history, and helper-library style analysis

## Search APIs

- `mlflow.search_runs()` is convenient when you want a DataFrame-oriented workflow
- `MlflowClient.search_runs()` returns `Run` objects, which is the preferred input for helper functions such as `runs_to_dataframe()`, `compare_configs()`, and `diagnose_run()`
