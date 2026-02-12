from pathlib import Path
from types import TracebackType
from typing import Optional

import pandas as pd
import yadisk


def delete_img_in_folder(folder_name):
    folder_name = Path(folder_name)
    if folder_name.exists():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".csv", ".pkl"}
        for p in folder_name.rglob("*"):
            if p.suffix.lower() in image_exts:
                p.unlink()


class CsvChunkDownloader:
    def __init__(
        self,
        file_csv,
        columns: list[str],
        yandex_token: Optional[str] = None,
        chunk_rows: int = 100,
        download_from_disk=False,
    ):
        self.file_csv = Path(file_csv)
        self.columns = columns
        self.buffer = []
        self.chunk_rows = chunk_rows
        self.yandex_token = yandex_token
        self.download_from_disk = download_from_disk
        if self.yandex_token is not None:
            self.client = yadisk.Client(token=self.yandex_token)

    def __enter__(self):
        if self.download_from_disk:
            remote_path = f"/{self.file_csv.name}"
            if self.client.exists(remote_path):
                print(f"Downloading existing CSV from Yandex.Disk: {remote_path}")
                self.client.download(remote_path, str(self.file_csv))
        print(f"Creating file {self.file_csv} and updating each {self.chunk_rows} rows")
        return self

    def update_csv(self, new_row: pd.Series):
        self.buffer.append(new_row.to_dict())

        if len(self.buffer) >= self.chunk_rows:
            self.upload_chunk()

    def upload_chunk(self):
        if not self.buffer:
            return

        df_chunk = pd.DataFrame(self.buffer, columns=self.columns)
        df_chunk.to_csv(
            self.file_csv,
            mode="a",
            header=not self.file_csv.exists(),
            index=False,
        )

        if self.yandex_token is not None:
            self.client.upload(
                str(self.file_csv), f"/{self.file_csv.name}", overwrite=True
            )

        self.buffer.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.upload_chunk()
        return False
