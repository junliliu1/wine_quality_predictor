"""
Download wine quality datasets from UCI Machine Learning Repository.

This script downloads the red and white wine quality datasets and saves them
to the specified output directory.

Usage:
    python scripts/01_download_data.py --output-dir data/raw
"""

import sys
from pathlib import Path

import click

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_download import download_wine_datasets, WINE_DATA_URLS


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/raw",
    help="Output directory for downloaded datasets",
)
@click.option(
    "--red-url",
    type=str,
    default=WINE_DATA_URLS["red"],
    help="URL for red wine dataset",
)
@click.option(
    "--white-url",
    type=str,
    default=WINE_DATA_URLS["white"],
    help="URL for white wine dataset",
)
def main(output_dir: str, red_url: str, white_url: str) -> None:
    """Download wine quality datasets from UCI repository."""
    print("=" * 60)
    print("STEP 1: DOWNLOAD DATA")
    print("=" * 60)

    red_path, white_path = download_wine_datasets(
        output_dir, red_url=red_url, white_url=white_url
    )

    print(f"  Downloaded: {red_path}")
    print(f"  Downloaded: {white_path}")
    print(f"\nDownload complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
