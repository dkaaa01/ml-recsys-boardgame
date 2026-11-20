"""Download a pinned BoardGameGeek Reviews snapshot from Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "jvanelteren/boardgamegeek-reviews?datasetVersionNumber={version}"
)
EXPECTED_MD5 = {
    3: "1319379ebcba3b0d3e7faa804e4caae6",
    4: "a4452b8e7216e714822d2a9306b14760",
}


def md5sum(path: Path) -> str:
    """Compute the archive fingerprint without loading it into memory.

    Parameters
    ----------
    path : Path
        Existing archive to read in 1 MiB chunks.

    Returns
    -------
    str
        Hexadecimal MD5 digest for identifying the pinned dataset bytes.
    """

    digest = hashlib.md5()  # noqa: S324 - integrity check, not security
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path, version: int = 4, force: bool = False) -> Path:
    """Download and verify a selected BoardGameGeek Reviews archive.

    Parameters
    ----------
    destination : Path
        Local archive path. Its parent directory is created when needed.
    version : int
        Kaggle dataset version, 3 or 4. Version 4 is the experiment snapshot.
    force : bool
        Replace an existing archive when True. Otherwise reuse it only when its
        fingerprint matches the requested version.

    Returns
    -------
    Path
        destination after the downloaded archive passes its MD5 check.

    The HTTP response is streamed to a .part file, then moved to destination.
    An unsupported version raises ValueError, and a mismatched fingerprint raises
    RuntimeError. Source dataset: https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews
    """

    if version not in EXPECTED_MD5:
        raise ValueError(f"Unsupported dataset version {version}; choose {sorted(EXPECTED_MD5)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if md5sum(destination) == EXPECTED_MD5[version]:
            print(f"Already downloaded and verified: {destination}")
            return destination
        raise RuntimeError(f"Existing file failed checksum: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        DATASET_URL.format(version=version), headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)

    actual = md5sum(destination)
    if actual != EXPECTED_MD5[version]:
        raise RuntimeError(
            f"Checksum mismatch: expected {EXPECTED_MD5[version]}, got {actual}"
        )
    print(f"Downloaded and verified: {destination}")
    return destination


def main() -> None:
    """Parse --output, --dataset-version and --force, then download the archive.

    The command accepts no Python arguments and returns None. --output is the
    archive path, --dataset-version chooses the pinned snapshot, and --force
    allows replacement of an existing file. argparse validates the version choice.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/boardgamegeek-reviews.zip"),
    )
    parser.add_argument("--dataset-version", type=int, default=4, choices=sorted(EXPECTED_MD5))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, args.dataset_version, args.force)


if __name__ == "__main__":
    main()

