"""src.new のテスト。"""

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

from src.new import _post_process, _resolve_code_sub, exp


@pytest.fixture
def template_dir(tmp_path: Path) -> Path:
    """テンプレートディレクトリを tmp_path に再現する。"""
    src = Path("templates/models")
    dest = tmp_path / "templates" / "models"
    shutil.copytree(src, dest)
    return dest


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


def _load_template_train_module(template_dir: Path, fake_mlflow: types.ModuleType) -> types.ModuleType:
    train_path = template_dir / "train.py"
    module_name = f"template_train_{id(train_path)}"
    original_sys_path = list(sys.path)
    original_mlflow = sys.modules.get("mlflow")
    original_settings = sys.modules.get("settings")
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(template_dir))
    sys.modules["mlflow"] = fake_mlflow
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_sys_path
        if original_mlflow is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = original_mlflow
        if original_settings is None:
            sys.modules.pop("settings", None)
        else:
            sys.modules["settings"] = original_settings


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    def start_run(self, *, run_name: str) -> _DummyRun:
        self.calls.append(("start_run", run_name))
        return _DummyRun()

    def log_params(self, params: dict[str, object]) -> None:
        self.calls.append(("log_params", params))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", tags))

    def log_metric(self, name: str, value: object) -> None:
        self.calls.append(("log_metric", (name, value)))


class TestResolveCodeSub:
    """_resolve_code_sub のテスト。"""

    def test_true(self) -> None:
        assert _resolve_code_sub("true") is True

    def test_false(self) -> None:
        assert _resolve_code_sub("false") is False

    def test_auto_kaggle_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPETITION_PLATFORM", "kaggle")
        monkeypatch.setenv("IS_CODE_COMPETITION", "true")
        assert _resolve_code_sub("auto") is True

    def test_auto_kaggle_non_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPETITION_PLATFORM", "kaggle")
        monkeypatch.setenv("IS_CODE_COMPETITION", "false")
        assert _resolve_code_sub("auto") is False

    def test_auto_non_kaggle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPETITION_PLATFORM", "signate")
        monkeypatch.setenv("IS_CODE_COMPETITION", "true")
        assert _resolve_code_sub("auto") is False

    def test_auto_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COMPETITION_PLATFORM", raising=False)
        monkeypatch.delenv("IS_CODE_COMPETITION", raising=False)
        assert _resolve_code_sub("auto") is False


class TestPostProcess:
    """_post_process のテスト。"""

    def test_template_replaces_exp_name(self, tmp_path: Path) -> None:
        """テンプレートから作成時、train.py の exp_name が置換される。"""
        target = tmp_path / "exp001"
        target.mkdir()
        (target / "train.py").write_text('settings = DirectorySettings(exp_name="template")\n')

        _post_process(target, "exp001", "template")

        train = (target / "train.py").read_text()
        assert 'exp_name="exp001"' in train
        assert 'exp_name="template"' not in train

    def test_source_replaces_exp_name(self, tmp_path: Path) -> None:
        """既存実験からコピー時、train.py の exp_name が置換される。"""
        target = tmp_path / "exp002"
        target.mkdir()
        (target / "train.py").write_text('settings = DirectorySettings(exp_name="exp001")\n')

        _post_process(target, "exp002", "exp001")

        train = (target / "train.py").read_text()
        assert 'exp_name="exp002"' in train

    def test_missing_train_py_no_error(self, tmp_path: Path) -> None:
        """train.py がなくてもエラーにならない。"""
        target = tmp_path / "exp001"
        target.mkdir()
        _post_process(target, "exp001", "template")


class TestExp:
    """exp コマンドのテスト。"""

    def test_create_from_template(self, template_dir: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """テンプレートから実験を作成できる。"""
        monkeypatch.chdir(template_dir.parent.parent)

        exp("exp001", source="template", kaggle_code_sub="false")

        target = models_dir / "exp001"
        assert target.exists()
        assert (target / "train.py").exists()
        assert not (target / "submission").exists()

        train = (target / "train.py").read_text()
        assert 'exp_name="exp001"' in train

    def test_create_with_kaggle_code_sub(
        self, template_dir: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kaggle_code_sub="true" の場合、submission ディレクトリが含まれる。"""
        monkeypatch.chdir(template_dir.parent.parent)

        exp("exp001", source="template", kaggle_code_sub="true")

        target = models_dir / "exp001"
        assert target.exists()
        assert (target / "submission").exists()

    def test_auto_includes_submission_for_kaggle_code_comp(
        self, template_dir: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto + kaggle + code competition の場合、submission が含まれる。"""
        monkeypatch.chdir(template_dir.parent.parent)
        monkeypatch.setenv("COMPETITION_PLATFORM", "kaggle")
        monkeypatch.setenv("IS_CODE_COMPETITION", "true")

        exp("exp001", source="template", kaggle_code_sub="auto")

        target = models_dir / "exp001"
        assert (target / "submission").exists()

    def test_auto_excludes_submission_for_non_code_comp(
        self, template_dir: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto + 非 code competition の場合、submission が除外される。"""
        monkeypatch.chdir(template_dir.parent.parent)
        monkeypatch.setenv("COMPETITION_PLATFORM", "kaggle")
        monkeypatch.setenv("IS_CODE_COMPETITION", "false")

        exp("exp001", source="template", kaggle_code_sub="auto")

        target = models_dir / "exp001"
        assert not (target / "submission").exists()

    def test_create_from_existing(self, template_dir: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """既存実験からコピーして作成できる。"""
        monkeypatch.chdir(template_dir.parent.parent)

        exp("exp001", source="template")
        exp("exp002", source="exp001")

        target = models_dir / "exp002"
        train = (target / "train.py").read_text()
        assert 'exp_name="exp002"' in train

    def test_already_exists_raises(self, template_dir: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG002
        """既に存在するディレクトリには作成できない。"""
        monkeypatch.chdir(template_dir.parent.parent)
        exp("exp001", source="template")

        with pytest.raises(AssertionError, match="Already exists"):
            exp("exp001", source="template")

    def test_source_not_found_raises(self, template_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """存在しないソースを指定するとエラーになる。"""
        monkeypatch.chdir(template_dir.parent.parent)

        with pytest.raises(AssertionError, match="Source not found"):
            exp("exp001", source="nonexistent")

    def test_template_train_skips_mlflow_when_uri_missing(
        self, template_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """MLFLOW_TRACKING_URI 未設定なら MLflow を使わず学習が完了する。"""
        fake_mlflow = types.ModuleType("mlflow")
        tracker = _FakeMlflow()
        fake_mlflow.set_tracking_uri = tracker.set_tracking_uri
        fake_mlflow.set_experiment = tracker.set_experiment
        fake_mlflow.start_run = tracker.start_run
        fake_mlflow.log_params = tracker.log_params
        fake_mlflow.set_tags = tracker.set_tags
        fake_mlflow.log_metric = tracker.log_metric
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COMPETITION_NAME", "demo-comp")
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        module = _load_template_train_module(template_dir, fake_mlflow)

        module.main(debug=True)

        assert tracker.calls == []

    def test_template_train_uses_mlflow_when_uri_present(
        self, template_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """MLFLOW_TRACKING_URI 設定時は MLflow の初期化と記録を行う。"""
        fake_mlflow = types.ModuleType("mlflow")
        tracker = _FakeMlflow()
        fake_mlflow.set_tracking_uri = tracker.set_tracking_uri
        fake_mlflow.set_experiment = tracker.set_experiment
        fake_mlflow.start_run = tracker.start_run
        fake_mlflow.log_params = tracker.log_params
        fake_mlflow.set_tags = tracker.set_tags
        fake_mlflow.log_metric = tracker.log_metric
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COMPETITION_NAME", "demo-comp")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        module = _load_template_train_module(template_dir, fake_mlflow)

        module.main(debug=False)

        call_names = [name for name, _payload in tracker.calls]
        assert call_names == [
            "set_tracking_uri",
            "set_experiment",
            "start_run",
            "log_params",
            "set_tags",
            "log_metric",
        ]
