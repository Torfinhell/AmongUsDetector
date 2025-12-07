from pytubefix import Playlist
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent


def download_videos(url, output_path):
    save_path = BASE / output_path
    save_path.mkdir(parents=True, exist_ok=True)
    pl = Playlist(url)
    for video in pl.videos:
        try:
            stream = video.streams.get_highest_resolution()
            filename = stream.default_filename
            full_path = os.path.join(save_path, filename)
            if os.path.exists(full_path):
                print(f"Skipping (already exists): {filename}")
                continue
            print(f"Downloading: {filename}")
            stream.download(save_path)
        except Exception as e:
            print(f"Error downloading video video: {video.title} with exception: {e}")


def main():
    url = "https://www.youtube.com/playlist?list=PLhjLcvbbPVrVOY5w5Pl7KuSe5fGroQJJD"
    save_path = "../data/videos"
    download_videos(url, save_path)


if __name__ == "__main__":
    main()
