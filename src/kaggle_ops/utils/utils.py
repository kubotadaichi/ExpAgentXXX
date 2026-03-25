import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()


def get_run_env() -> str:
    if os.getenv("KAGGLE_DATA_PROXY_TOKEN"):
        return "kaggle"
    if os.getenv("BUCKET_NAME") and Path(f"/gcs/{os.getenv('BUCKET_NAME')}").exists():
        return "vertex"
    return "local"


def build_kaggle_api():
    from kaggle import KaggleApi

    client = KaggleApi()
    client.authenticate()
    return client


def get_kaggle_username(*, required: bool = False) -> str:
    username = os.getenv("KAGGLE_USERNAME", "")
    if required and not username:
        raise KeyError("KAGGLE_USERNAME is required")
    return username


def get_kaggle_auth_env() -> dict[str, str]:
    api_token = os.getenv("KAGGLE_API_TOKEN", "")
    if api_token:
        env = {"KAGGLE_API_TOKEN": api_token}
        username = get_kaggle_username()
        if username:
            env["KAGGLE_USERNAME"] = username
        return env

    username = get_kaggle_username(required=True)
    key = os.getenv("KAGGLE_KEY", "")
    if not key:
        raise KeyError("KAGGLE_KEY is required when KAGGLE_API_TOKEN is not set")
    return {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
