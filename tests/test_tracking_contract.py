from pathlib import Path
import json
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_uses_mlflow_not_wandb() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert any(dep.startswith("mlflow") for dep in deps)
    assert not any(dep.startswith("wandb") for dep in deps)


def test_env_example_uses_mlflow_tracking_uri() -> None:
    text = (ROOT / ".env.example").read_text()
    assert "MLFLOW_TRACKING_URI=" in text
    assert "WANDB_API_KEY=" not in text


def test_readme_mentions_mlflow_setup() -> None:
    text = (ROOT / "README.md").read_text()
    assert "MLFLOW_TRACKING_URI=" in text
    assert "wandb/skills" not in text
    assert "mlflow" in text.lower()


def test_skills_lock_does_not_reference_wandb_primary() -> None:
    data = json.loads((ROOT / "skills-lock.json").read_text())
    assert "wandb-primary" not in data["skills"]
