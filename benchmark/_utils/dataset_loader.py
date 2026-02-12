from pathlib import Path
from typing import List


def get_dataset_files(directory: str) -> List[str]:
    """
    Returns list of absolute paths to .g6 files in the directory.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    return sorted([str(p.absolute()) for p in path.glob("*.g6")])
