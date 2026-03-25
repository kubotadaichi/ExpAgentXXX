from pathlib import Path

from src.kaggle_ops import vertex


class DummyJob:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs


def test_train_skips_mlflow_tracking_uri_when_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECT_ID", "demo-project")
    monkeypatch.setenv("REGION", "asia-northeast1")
    monkeypatch.setenv("COMPETITION_NAME", "demo-comp")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["mlflow>=2.0.0"]\n')

    holder = {}

    def fake_compile_train_script(*, exp: str) -> str:
        assert exp == "exp001"
        return str(tmp_path / "compiled.py")

    monkeypatch.setattr(vertex, "compile_train_script", fake_compile_train_script)
    monkeypatch.setattr(vertex, "_upload_to_gcs", lambda *_args: "gs://bucket/scripts/compiled.py")
    monkeypatch.setattr(vertex.aiplatform, "init", lambda **_kwargs: None)

    def fake_job(**kwargs):
        job = DummyJob(**kwargs)
        holder["job"] = job
        return job

    monkeypatch.setattr(vertex.aiplatform, "CustomContainerTrainingJob", fake_job)

    vertex.train("exp001")

    env_vars = holder["job"].run_kwargs["environment_variables"]
    assert "MLFLOW_TRACKING_URI" not in env_vars
    assert "mlflow>=2.0.0" in env_vars["REQUIREMENTS"]


def test_train_passes_mlflow_tracking_uri_when_set(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECT_ID", "demo-project")
    monkeypatch.setenv("REGION", "asia-northeast1")
    monkeypatch.setenv("COMPETITION_NAME", "demo-comp")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")

    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["mlflow>=2.0.0"]\n')

    holder = {}

    def fake_compile_train_script(*, exp: str) -> str:
        assert exp == "exp001"
        return str(tmp_path / "compiled.py")

    monkeypatch.setattr(vertex, "compile_train_script", fake_compile_train_script)
    monkeypatch.setattr(vertex, "_upload_to_gcs", lambda *_args: "gs://bucket/scripts/compiled.py")
    monkeypatch.setattr(vertex.aiplatform, "init", lambda **_kwargs: None)

    def fake_job(**kwargs):
        job = DummyJob(**kwargs)
        holder["job"] = job
        return job

    monkeypatch.setattr(vertex.aiplatform, "CustomContainerTrainingJob", fake_job)

    vertex.train("exp001")

    env_vars = holder["job"].run_kwargs["environment_variables"]
    assert env_vars["MLFLOW_TRACKING_URI"] == "http://mlflow.internal:5000"
    assert "mlflow>=2.0.0" in env_vars["REQUIREMENTS"]
