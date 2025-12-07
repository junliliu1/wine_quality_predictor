"""
Download wine quality datasets from UCI Machine Learning Repository.

Usage: python scripts/01_download_data.py --output-dir data/raw
"""

import click
import requests
from pathlib import Path

WINE_DATA_URLS = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
}


def download_file(url: str, output_path: Path) -> None:
    """Download file from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"  Downloaded: {output_path}")


@click.command()
@click.option("--output-dir", type=click.Path(), default="data/raw", help="Output directory")
@click.option("--red-url", type=str, default=WINE_DATA_URLS["red"], help="Red wine URL")
@click.option("--white-url", type=str, default=WINE_DATA_URLS["white"], help="White wine URL")
def main(output_dir: str, red_url: str, white_url: str) -> None:
    """Download wine datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: DOWNLOAD DATA")
    print("=" * 60)

    download_file(red_url, output_path / "winequality-red.csv")
    download_file(white_url, output_path / "winequality-white.csv")

    print(f"\nDownload complete! Files saved to: {output_path}")


if __name__ == "__main__":
    main()
