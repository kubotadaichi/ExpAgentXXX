import logging
import os

import dotenv
from tyro.extras import SubcommandApp

from .utils.customhub import competition_download, datasets_download
from .utils.utils import build_kaggle_api

dotenv.load_dotenv()
app = SubcommandApp()
logger = logging.getLogger(__name__)
INPUT_DIR = "./data/input"


@app.command()
def competition_dataset(force_download: bool = False) -> None:
    competition_download(
        client=build_kaggle_api(),
        handle=os.environ["COMPETITION_NAME"],
        destination=INPUT_DIR,
        force_download=force_download,
    )


@app.command()
def datasets(handles: list[str], force_download: bool = False) -> None:
    datasets_download(
        client=build_kaggle_api(),
        handles=handles,
        destination=INPUT_DIR,
        force_download=force_download,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    app.cli()
