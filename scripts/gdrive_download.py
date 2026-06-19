import os
import shutil
import zipfile
from pathlib import Path

import gdown
import kagglehub

GDRIVE_URLS = {
    "models": {},
    "dataset": {
        "https://drive.google.com/uc?id=10HMuMEpgW0JaTyPdmMhsnmzg5O3R_Ex8": "data/annotations",
        # "https://drive.google.com/uc?id=1oSWcgPzh6TEFNpBzw4qEvEifx2s3wBKA": "data",
        "https://drive.google.com/uc?id=1wEn7jSji4rp-xfUM5gvaL-e6gN_DYX9K": "data",
        "https://drive.google.com/uc?id=1LzDgHcTxwoJpMPb09gDnnwcOteKkexDw": "data",
        "https://drive.google.com/uc?id=1LHudhuGDHhXd6pJABSsw_rVOo1px3hZM": "data",
    },
}


def download_checkpoints(gdrive_urls):
    if "models" not in gdrive_urls:
        raise ValueError("Cannot upload model files")
    for url, path in gdrive_urls["models"].items():
        Path(path).mkdir(exist_ok=True, parents=True)
        zip_path = path + ".zip"
        gdown.download(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)
        Path(zip_path).unlink()


def download_dataset(gdrive_urls):
    if "dataset" not in gdrive_urls:
        raise ValueError("Cannot upload dataset files")

    for url, target_dir_str in gdrive_urls["dataset"].items():
        target_dir = Path(target_dir_str)
        target_dir.mkdir(exist_ok=True, parents=True)

        downloaded_file_str = gdown.download(url, output=None, quiet=False)
        if not downloaded_file_str:
            continue

        downloaded_path = Path(downloaded_file_str)

        if zipfile.is_zipfile(downloaded_path):
            with zipfile.ZipFile(downloaded_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            downloaded_path.unlink()
        else:
            final_path = target_dir / downloaded_path.name
            shutil.move(str(downloaded_path), str(final_path))


def download_dataset_kaggle():
    downloaded_path = kagglehub.dataset_download("nikitasolonitsyn/among_us_detection")
    target_path = "./data"
    if not os.path.exists(target_path):
        shutil.copytree(downloaded_path, target_path)
    print("Dataset available at:", target_path)


if __name__ == "__main__":
    # download_checkpoints(GDRIVE_URLS)
    # download_dataset(GDRIVE_URLS)
    download_dataset_kaggle()  # TODO check this
