from pathlib import Path
from typing import Optional

import pandas as pd
import yadisk


def delete_img_in_folder(folder_name: str | Path):
    folder_path = Path(folder_name)
    if folder_path.exists():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".csv", ".pkl"}
        for p in folder_path.rglob("*"):
            if p.suffix.lower() in valid_exts:
                p.unlink()


class CsvChunkDownloader:
    def __init__(
        self,
        file_csv: str | Path,
        columns: list[str],
        yandex_token: Optional[str] = None,
        chunk_rows: Optional[int] = 100,
        download_from_disk: bool = False,
    ):
        self.file_csv = Path(file_csv)
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.download_from_disk = download_from_disk
        self.buffer = []
        self._header_written = self.file_csv.exists()

        self.remote_path = f"/{self.file_csv.name}"
        self.client = yadisk.Client(token=yandex_token) if yandex_token else None

        # Initial synchronization down from remote cloud storage
        if self.client and self.download_from_disk:
            self._download_if_exists()

    def __enter__(self):
        return self

    def _download_if_exists(self):
        """Internal helper to pull the latest file state from Yandex.Disk."""
        if self.client.exists(self.remote_path):
            print(f"Downloading existing CSV from Yandex.Disk: {self.remote_path}")
            self.client.download(self.remote_path, str(self.file_csv))
            self._header_written = True

    def update_csv(self, new_row: pd.Series):
        self.buffer.append(new_row.to_dict())
        if self.chunk_rows is not None and len(self.buffer) >= self.chunk_rows:
            self.upload_chunk()

    def upload_chunk(self):
        if not self.buffer:
            return

        # Sync right before writing to preserve concurrent multi-node adjustments
        if self.client and self.download_from_disk:
            self._download_if_exists()

        df_chunk = pd.DataFrame(self.buffer, columns=self.columns)
        df_chunk.to_csv(
            self.file_csv,
            mode="a",
            header=not self._header_written,
            index=False,
        )
        self._header_written = True

        if self.client:
            self.client.upload(str(self.file_csv), self.remote_path, overwrite=True)

        self.buffer.clear()

    def get_csv(self, default_columns: list[str]) -> pd.DataFrame:
        if not self.file_csv.exists():
            return pd.DataFrame(columns=default_columns)
        return pd.read_csv(self.file_csv)

    def __exit__(self, exc_type, exc_value, traceback):
        self.upload_chunk()
        if self.client:
            self.client.close()  # Properly close the network session
        return False
