import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.kaggle_ops.utils.customhub",
        "src.kaggle_ops.download",
        "src.kaggle_ops.upload",
    ],
)
def test_kaggle_modules_do_not_require_auth_on_import(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kaggle 関連モジュールは import 時に認証情報を要求しない。"""
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(key, raising=False)

    for key in list(sys.modules):
        if key == module_name or key.startswith(f"{module_name}."):
            sys.modules.pop(key, None)

    importlib.import_module(module_name)
