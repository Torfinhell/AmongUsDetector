import zipfile
from pathlib import Path

import gdown
import kagglehub

GDRIVE_URLS = {
    "models": {},
    "dataset": {
        "https://drive.google.com/uc?id=10HMuMEpgW0JaTyPdmMhsnmzg5O3R_Ex8": "data/annotations",
        "https://drive.google.com/uc?id=1oSWcgPzh6TEFNpBzw4qEvEifx2s3wBKA": "data",
        "https://drive.google.com/uc?id=1wEn7jSji4rp-xfUM5gvaL-e6gN_DYX9K": "data",
        "https://drive.google.com/uc?id=1LzDgHcTxwoJpMPb09gDnnwcOteKkexDw": "data",
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
    for url, path in gdrive_urls["dataset"].items():
        Path(path).mkdir(exist_ok=True, parents=True)
        zip_path = path + ".zip"
        gdown.download(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)
        Path(zip_path).unlink()


def download_dataset_kaggle():
    dataset_path = kagglehub.dataset_download(
        "nikitasolonitsyn/among_us_detection", output_dir="data"
    )
    dataset_path = f"data/{dataset_path}"
    print("Dataset downloaded to:", dataset_path)


if __name__ == "__main__":
    download_checkpoints(GDRIVE_URLS)
    download_dataset(GDRIVE_URLS)
    # download_dataset_kaggle()
