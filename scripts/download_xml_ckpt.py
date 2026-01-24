"https://drive.google.com/file/d/1C3_CgNdPWGrTTacLsRCLPS1bU40Be8hU/view?usp=sharing"
"https://drive.google.com/file/d/1qbtMWIQQ2ym1f13GkmdQeR67fzfpX7nA/view?usp=sharing"
from pathlib import Path
import gdown
import zipfile

GDRIVE_URLS = {
    "models": {
    },
    "dataset": {
        "https://drive.google.com/uc?id=1C3_CgNdPWGrTTacLsRCLPS1bU40Be8hU": "data/annotations",
        "https://drive.google.com/uc?id=1qbtMWIQQ2ym1f13GkmdQeR67fzfpX7nA": "data"
    }
}

def download_checkpoints(gdrive_urls):
    if "models" not in gdrive_urls:
        raise ValueError("Cannot upload model files")
    path_gzip = Path("data/models").absolute()
    path_gzip.mkdir(exist_ok=True, parents=True)
    for url, path in gdrive_urls["models"].items():
        gdown.download(url, path)

def download_dataset(gdrive_urls):
    if "dataset" not in gdrive_urls:
        raise ValueError("Cannot upload dataset files")
    for url, path in gdrive_urls["dataset"].items():
        Path(path).mkdir(exist_ok=True, parents=True)
        zip_path = path + ".zip"
        gdown.download(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        Path(zip_path).unlink() 

if __name__ == "__main__":
    download_checkpoints(GDRIVE_URLS)
    download_dataset(GDRIVE_URLS)