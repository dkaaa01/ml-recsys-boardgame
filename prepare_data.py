"""Download the files used in the 20,000-user experiment.

Source: https://github.com/dkaaa01/ml-recsys-boardgame
The commit below fixes the file versions used in the submitted results.
The small recommendation sample is already included in the submission.
"""

import shutil
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
COMMIT = "19b049605b7b34fa9689dc83df5adec5181e1922"
BASE_URL = f"https://raw.githubusercontent.com/dkaaa01/ml-recsys-boardgame/{COMMIT}/"
FILES = (
    "data/experiment/games.csv",
    "data/experiment/users.csv",
    "data/experiment/train.csv",
    "data/experiment/validation.csv",
    "data/experiment/test.csv",
    "data/experiment/eval_users.csv",
    "data/experiment/popularity.csv",
    "outputs/top10_test.csv",
)


def main() -> None:
    """Download missing experiment CSVs to the paths expected by the code.

    No arguments are required. Existing files are kept. urllib.request.urlopen
    reads each pinned URL with a 60-second timeout, and shutil.copyfileobj
    streams the response to disk. A temporary file is renamed only after the
    transfer finishes, so an interrupted download can be retried.
    """
    for relative_path in FILES:
        target = ROOT / relative_path
        if target.exists():
            print(f"Already present: {relative_path}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        try:
            with urlopen(BASE_URL + relative_path, timeout=60) as source:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
            temporary.replace(target)
        finally:
            # missing_ok also covers a transfer that failed before file creation.
            temporary.unlink(missing_ok=True)
        print(f"Downloaded: {relative_path}")


if __name__ == "__main__":
    main()
