"""Download the image search results to data/construction/document_images/<uuid>.jpg.

Each file is named by generate_random_id(link). The full-size link is tried first and the
Google thumbnail is used when it fails. Images that cannot be opened are deleted afterwards.

    python -m data_construction.download_images
"""
import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from util import generate_random_id

MAX_TIME = 5
MAX_RETRY = 2


def fetch(url, save_path):
    subprocess.run(["wget", "-q", f"--timeout={MAX_TIME}", f"--tries={MAX_RETRY}", url, "-O", save_path], check=False)
    return os.path.isfile(save_path) and os.path.getsize(save_path) > 0


def is_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download(link, thumbnail, save_dir):
    save_path = os.path.join(save_dir, f"{generate_random_id(link)}.jpg")
    if os.path.isfile(save_path):
        return
    for url in (link.replace("http://", "https://"), thumbnail):
        if url and fetch(url, save_path) and is_valid(save_path):
            return
    if os.path.exists(save_path):
        os.remove(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_dir", type=str, default="data/construction/search")
    parser.add_argument("--save_dir", type=str, default="data/construction/document_images")
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    links = {}
    for path in glob.glob(os.path.join(args.search_dir, "search_results_image_*.json")):
        for results in json.load(open(path)):
            for result in results:
                if "redd" not in result["link"]:
                    links[result["link"]] = result.get("image", {}).get("thumbnailLink")
    print(f"{len(links)} unique image links")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, link, thumb, args.save_dir) for link, thumb in links.items()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
