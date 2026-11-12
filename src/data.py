"""NASA C-MAPSS FD001 download, validation, and loading utilities.

Dataset source
--------------
Saxena, A. & Goebel, K. (2008). C-MAPSS Turbofan Engine Degradation Simulation
Data Set. NASA Prognostics Center of Excellence Data Repository.
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

BASE_COLUMNS = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SUPPORTED_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}


# ---------------------------------------------------------------------------
# Download & Extraction
# ---------------------------------------------------------------------------

def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
    """Extract all members, refusing Zip-Slip path traversal attacks."""
    target_dir = target_dir.resolve()
    for member in zf.infolist():
        dest = (target_dir / member.filename).resolve()
        if dest == target_dir:
            continue
        if target_dir not in dest.parents:
            raise ValueError(f"Unsafe archive member detected: {member.filename}")
    zf.extractall(target_dir)


def _locate_cmapss_dir(search_root: Path) -> Optional[Path]:
    """Walk *search_root* and return the directory holding train_FD001.txt."""
    for candidate in search_root.rglob("train_FD001.txt"):
        return candidate.parent
    return None


def download_fd001(force: bool = False) -> Path:
    """Download the NASA outer archive, extract the inner CMAPSSData zip, and
    return the directory that contains the .txt data files.

    Parameters
    ----------
    force:
        Re-download even if data files already exist locally.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: data already present.
    cmapss_dir = _locate_cmapss_dir(RAW_DIR)
    if cmapss_dir is not None and not force:
        return cmapss_dir

    outer_archive = RAW_DIR / "nasa_turbofan_outer.zip"
    print(f"Downloading NASA C-MAPSS archive from {DATASET_URL} …")
    urllib.request.urlretrieve(DATASET_URL, outer_archive)
    print("Download complete. Extracting …")

    with zipfile.ZipFile(outer_archive) as outer_zf:
        # The outer zip normally contains a CMAPSSData.zip inside (possibly in
        # a subdirectory).  Find it and unpack in-memory.
        inner_name = next(
            (m for m in outer_zf.namelist() if Path(m).name == "CMAPSSData.zip"),
            None,
        )
        if inner_name is not None:
            inner_bytes = outer_zf.read(inner_name)
            with zipfile.ZipFile(BytesIO(inner_bytes)) as inner_zf:
                cmapss_target = RAW_DIR / "CMAPSSData"
                cmapss_target.mkdir(exist_ok=True)
                _safe_extract_zip(inner_zf, cmapss_target)
        else:
            # Some versions ship without a nested zip — extract directly.
            _safe_extract_zip(outer_zf, RAW_DIR)

    cmapss_dir = _locate_cmapss_dir(RAW_DIR)
    if cmapss_dir is None:
        raise FileNotFoundError(
            "Extraction succeeded but train_FD001.txt was not found under "
            f"{RAW_DIR}.  Inspect the archive manually."
        )
    print(f"C-MAPSS files extracted to: {cmapss_dir}")
    return cmapss_dir


def _dataset_dir() -> Path:
    """Return the directory holding C-MAPSS .txt files; raise if absent."""
    cmapss_dir = _locate_cmapss_dir(RAW_DIR)
    if cmapss_dir is not None:
        return cmapss_dir
    raise FileNotFoundError(
        "FD001 data not found.  Run:\n"
        "    python -m src.data --download\n"
        "to fetch the NASA C-MAPSS dataset."
    )


# ---------------------------------------------------------------------------
# Parsing & Validation
# ---------------------------------------------------------------------------

def read_trajectory(path: Path) -> pd.DataFrame:
    """Parse a whitespace-delimited C-MAPSS trajectory file into a DataFrame.

    Some versions of the archive append two all-NaN trailing columns; these are
    silently dropped before column assignment.
    """
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Drop trailing all-NaN columns (artefact of some archive versions).
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != len(BASE_COLUMNS):
        raise ValueError(
            f"Expected {len(BASE_COLUMNS)} columns in {path.name}; "
            f"found {frame.shape[1]}."
        )
    frame.columns = BASE_COLUMNS
    return frame


def validate_trajectory(frame: pd.DataFrame) -> None:
    """Raise a descriptive ValueError when *frame* violates C-MAPSS invariants.

    Checks performed
    ----------------
    * All BASE_COLUMNS are present.
    * No entirely-empty usable data.
    * engine_id and cycle are strictly positive integers.
    * No duplicate (engine_id, cycle) pairs.
    """
    missing = set(BASE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame[BASE_COLUMNS].isna().all(axis=None):
        raise ValueError("Trajectory is empty or contains no usable measurements.")
    if (frame["engine_id"] <= 0).any() or (frame["cycle"] <= 0).any():
        raise ValueError("engine_id and cycle must be strictly positive integers.")
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(
            "Each (engine_id, cycle) observation must be unique.  "
            "Found duplicate rows."
        )


def add_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Append a row-wise RUL column derived from each engine's max observed cycle.

    RUL = max_cycle_for_engine – current_cycle.  The final row of each engine
    therefore has RUL = 0.
    """
    validate_trajectory(frame)
    result = frame.copy()
    result["RUL"] = (
        result.groupby("engine_id")["cycle"].transform("max") - result["cycle"]
    )
    return result


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_train(subset: str = "FD001") -> pd.DataFrame:
    """Load a complete run-to-failure training partition with computed RUL.

    Parameters
    ----------
    subset:
        One of ``'FD001'`` through ``'FD004'``.  Only FD001 is officially
        supported by this project.
    """
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"subset must be one of {SUPPORTED_SUBSETS}.")
    frame = read_trajectory(_dataset_dir() / f"train_{subset}.txt")
    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return add_rul(frame)


def load_test(subset: str = "FD001") -> tuple[pd.DataFrame, np.ndarray]:
    """Load truncated test trajectories and the official end-of-trajectory RUL labels.

    Returns
    -------
    frame:
        All test observations.
    y:
        1-D array of official RUL values, one per engine, ordered by engine ID
        (1, 2, … N).
    """
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"subset must be one of {SUPPORTED_SUBSETS}.")
    directory = _dataset_dir()
    frame = (
        read_trajectory(directory / f"test_{subset}.txt")
        .sort_values(["engine_id", "cycle"])
        .reset_index(drop=True)
    )
    labels = (
        pd.read_csv(
            directory / f"RUL_{subset}.txt",
            sep=r"\s+",
            header=None,
            engine="python",
        )
        .dropna(axis=1, how="all")
    )
    if labels.shape[1] != 1:
        raise ValueError(
            "Official RUL file must contain exactly one label column per engine."
        )
    y = labels.iloc[:, 0].to_numpy(dtype=float)
    n_engines = frame["engine_id"].nunique()
    if len(y) != n_engines:
        raise ValueError(
            f"Official RUL file contains {len(y)} labels but {n_engines} test "
            "engines were found."
        )
    return frame, y


# ---------------------------------------------------------------------------
# Processed data persistence
# ---------------------------------------------------------------------------

def save_processed(subset: str = "FD001") -> tuple[Path, Path]:
    """Write cleaned, labelled train and clean test frames to data/processed/.

    Returns the two written file paths.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / f"train_{subset}.csv"
    test_path = PROCESSED_DIR / f"test_{subset}.csv"
    load_train(subset).to_csv(train_path, index=False)
    test_frame, test_labels = load_test(subset)
    test_frame.to_csv(test_path, index=False)
    labels_path = PROCESSED_DIR / f"RUL_{subset}.csv"
    pd.DataFrame({"RUL": test_labels}).to_csv(labels_path, index=False)
    return train_path, test_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess NASA C-MAPSS FD001."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the NASA archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data files are already present.",
    )
    parser.add_argument(
        "--subset",
        default="FD001",
        choices=list(SUPPORTED_SUBSETS),
        help="Which C-MAPSS subset to process (default: FD001).",
    )
    args = parser.parse_args()
    if args.download:
        download_fd001(force=args.force)
    paths = save_processed(args.subset)
    print("Saved processed files:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
