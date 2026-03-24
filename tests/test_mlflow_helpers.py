import importlib.util
import types
from pathlib import Path

HELPERS_PATH = Path(".claude/skills/mlflow-primary/scripts/mlflow_helpers.py")


def load_helpers():
    spec = importlib.util.spec_from_file_location("mlflow_helpers", HELPERS_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_run(run_id: str, name: str, params: dict, metrics: dict):
    return types.SimpleNamespace(
        info=types.SimpleNamespace(run_id=run_id, experiment_id="1", status="FINISHED", start_time=123),
        data=types.SimpleNamespace(params=params, metrics=metrics, tags={"mlflow.runName": name}),
    )


def test_runs_to_dataframe_flattens_runs() -> None:
    helpers = load_helpers()
    rows = helpers.runs_to_dataframe(
        [fake_run("r1", "baseline", {"lr": "0.1"}, {"loss": 0.4})],
        metric_keys=["loss"],
    )
    assert rows == [
        {
            "run_id": "r1",
            "run_name": "baseline",
            "status": "FINISHED",
            "experiment_id": "1",
            "start_time": 123,
            "param.lr": "0.1",
            "loss": 0.4,
        }
    ]


def test_compare_configs_only_returns_differences() -> None:
    helpers = load_helpers()
    left = fake_run("r1", "baseline", {"lr": "0.1", "depth": "6"}, {})
    right = fake_run("r2", "trial", {"lr": "0.2", "depth": "6"}, {})
    diffs = helpers.compare_configs(left, right)
    assert diffs == [{"key": "lr", "baseline": "0.1", "trial": "0.2"}]
