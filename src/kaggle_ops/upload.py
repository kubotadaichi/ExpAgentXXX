import logging
import os
from pathlib import Path

import dotenv
from tyro.extras import SubcommandApp

from .utils.customhub import model_upload
from .utils.utils import build_kaggle_api, get_kaggle_username

dotenv.load_dotenv()
app = SubcommandApp()
logger = logging.getLogger(__name__)


@app.command()
def models(exp: str) -> None:
    model_upload(
        client=build_kaggle_api(),
        handle=f"{get_kaggle_username(required=True)}/{os.environ['COMPETITION_NAME']}-models/other/{exp}",
        local_model_dir=str(Path(f"models/{exp}")),
        update=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    app.cli()
