"""Download the fixed BoardGameGeek Reviews v3 snapshot from Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "jvanelteren/boardgamegeek-reviews?datasetVersionNumber=3"
)
EXPECTED_MD5 = "1319379ebcba3b0d3e7faa804e4caae6"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check, not security
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if md5sum(destination) == EXPECTED_MD5:
            print(f"Already downloaded and verified: {destination}")
            return destination
        raise RuntimeError(f"Existing file failed checksum: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)

    actual = md5sum(destination)
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"Checksum mismatch: expected {EXPECTED_MD5}, got {actual}")
    print(f"Downloaded and verified: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/boardgamegeek-reviews-v3.zip"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, args.force)


if __name__ == "__main__":
    main()

