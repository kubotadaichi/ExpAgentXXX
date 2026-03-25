import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_uses_mlflow_not_wandb() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert any(dep.startswith("mlflow") for dep in deps)
    assert not any(dep.startswith("wandb") for dep in deps)


def test_env_example_does_not_force_mlflow_uri() -> None:
    text = (ROOT / ".env.example").read_text()
    assert "MLFLOW_TRACKING_URI=" in text
    assert "MLFLOW_TRACKING_URI=http://localhost:5000" not in text


def test_readme_describes_mlflow_as_optional() -> None:
    text = (ROOT / "README.md").read_text()
    assert "Leave `MLFLOW_TRACKING_URI` empty" in text
    assert "MLflow logging is optional" in text


def test_readme_keeps_mlflow_server_explanation_minimal() -> None:
    text = (ROOT / "README.md").read_text()
    assert "outside this repository" in text


def test_taskfile_does_not_manage_mlflow_server() -> None:
    text = (ROOT / "Taskfile.yml").read_text()
    assert "mlflow-server:" not in text
