import os
from contextlib import nullcontext
from typing import Any, cast

import mlflow

from settings import Config, DirectorySettings

mlflow = cast(Any, mlflow)


def _start_mlflow_run(settings: DirectorySettings, debug: bool):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        return nullcontext()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.competition_name)
    run_name = f"{settings.exp_name}-debug" if debug else settings.exp_name
    return mlflow.start_run(run_name=run_name)


def main(debug: bool = False) -> None:
    """学習スクリプト。--debug で動作確認用の短縮実行。"""
    settings = DirectorySettings(exp_name="template")
    config = Config()
    tracking_enabled = bool(os.getenv("MLFLOW_TRACKING_URI", "").strip())

    if debug:
        settings.artifact_dir = settings.artifact_dir / "debug"
        settings.output_dir = settings.artifact_dir
        config.epochs = 1
        # config.max_length = 128

    print(f"debug: {debug}")
    print(f"run_env: {settings.run_env}")
    print(f"comp_dataset_dir: {settings.comp_dataset_dir}")
    print(f"output_dir: {settings.output_dir}")
    print(f"artifact_dir: {settings.artifact_dir}")
    print(f"config: {config.model_dump()}")

    # TODO: Implement training logic

    # if debug:
    #     train_df = train_df.head(100)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)

    with _start_mlflow_run(settings, debug):
        if tracking_enabled:
            mlflow.log_params(config.model_dump())
            mlflow.set_tags(
                {
                    "exp_name": settings.exp_name,
                    "run_env": settings.run_env,
                    "debug": str(debug).lower(),
                }
            )
            mlflow.log_metric("completed", 1)

        print("Training completed!")

    # TODO: バリデーション推論
    # val_predictions = predict(model, tokenizer, val_df, max_length=config.max_length)
    # chrF / BLEU / geo_mean を計算し、OOF予測をCSVに保存


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
