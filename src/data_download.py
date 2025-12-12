"""
Data download utilities for wine quality prediction.

This module provides functions to download wine quality datasets from the
UCI Machine Learning Repository.

Functions:
    download_file: Download a single file from a URL.
    download_wine_datasets: Download both red and white wine datasets.
"""

from pathlib import Path
from typing import Union

import requests


# Default URLs for UCI wine quality datasets
WINE_DATA_URLS = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}


def download_file(url: str, output_path: Union[str, Path], timeout: int = 30) -> Path:
    """
    Download a file from a URL and save it to the specified path.

    Parameters
    ----------
    url : str
        The URL to download from.
    output_path : str or Path
        The local file path where the downloaded content will be saved.
        Parent directories will be created if they don't exist.
    timeout : int, optional
        Request timeout in seconds (default: 30).

    Returns
    -------
    Path
        The path to the downloaded file.

    Raises
    ------
    ValueError
        If url is empty or output_path is empty.
    requests.exceptions.RequestException
        If the download fails (network error, HTTP error, etc.).

    Examples
    --------
    >>> download_file("https://example.com/data.csv", "data/data.csv")
    PosixPath('data/data.csv')
    """
    if not url:
        raise ValueError("URL cannot be empty")
    if not output_path:
        raise ValueError("Output path cannot be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return output_path


def download_wine_datasets(
    output_dir: Union[str, Path],
    red_url: str = WINE_DATA_URLS["red"],
    white_url: str = WINE_DATA_URLS["white"],
    timeout: int = 30,
) -> tuple[Path, Path]:
    """
    Download both red and white wine quality datasets.

    Parameters
    ----------
    output_dir : str or Path
        Directory where the datasets will be saved.
    red_url : str, optional
        URL for red wine dataset (default: UCI repository URL).
    white_url : str, optional
        URL for white wine dataset (default: UCI repository URL).
    timeout : int, optional
        Request timeout in seconds (default: 30).

    Returns
    -------
    tuple[Path, Path]
        Tuple containing paths to (red_wine_file, white_wine_file).

    Raises
    ------
    ValueError
        If output_dir is empty.
    requests.exceptions.RequestException
        If either download fails.

    Examples
    --------
    >>> red_path, white_path = download_wine_datasets("data/raw")
    >>> red_path
    PosixPath('data/raw/winequality-red.csv')
    """
    if not output_dir:
        raise ValueError("Output directory cannot be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    red_file = download_file(red_url, output_path / "winequality-red.csv", timeout)
    white_file = download_file(white_url, output_path / "winequality-white.csv", timeout)

    return red_file, white_file
