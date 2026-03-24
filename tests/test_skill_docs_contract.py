from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mlflow_primary_skill_exists() -> None:
    text = (ROOT / ".claude/skills/mlflow-primary/SKILL.md").read_text()
    assert "mlflow-primary" in text
    assert "MLFLOW_SDK.md" in text
    assert "mlflow_helpers.py" in text


def test_experiment_workflow_points_to_mlflow() -> None:
    text = (ROOT / ".claude/skills/experiment-workflow/SKILL.md").read_text()
    assert "mlflow-primary" in text
    assert "wandb-primary" not in text
    assert "Weights & Biases" not in text


def test_archived_weave_files_exist_under_mlflow_skill() -> None:
    assert (ROOT / ".claude/skills/mlflow-primary/references/_WEAVE_SDK.md").exists()
    assert (ROOT / ".claude/skills/mlflow-primary/scripts/_weave_helpers.py").exists()
