"""
Clean and transform wine quality data.

This script loads red and white wine datasets, validates them using Pandera,
removes duplicates, and creates quality categories for classification.

Usage:
    python scripts/02_clean_data.py --red-wine data/raw/winequality-red.csv \
        --white-wine data/raw/winequality-white.csv \
        --output-path data/processed/wine_data_cleaned.csv
"""

import sys
from pathlib import Path

import click

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import (
    load_and_combine_data,
    validate_wine_data,
    clean_wine_data,
    EXPECTED_COLUMNS,
)


@click.command()
@click.option(
    "--red-wine",
    type=click.Path(exists=True),
    required=True,
    help="Path to red wine CSV file",
)
@click.option(
    "--white-wine",
    type=click.Path(exists=True),
    required=True,
    help="Path to white wine CSV file",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default="data/processed/wine_data_cleaned.csv",
    help="Output path for cleaned data",
)
def main(red_wine: str, white_wine: str, output_path: str) -> None:
    """Clean and transform wine data."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 2: CLEAN/TRANSFORM DATA")
    print("=" * 60)

    print("\n1. Loading data...")
    wine_data = load_and_combine_data(Path(red_wine), Path(white_wine))
    red_count = len(wine_data[wine_data["wine_type"] == 0])
    white_count = len(wine_data[wine_data["wine_type"] == 1])
    print(f"  Red: {red_count}, White: {white_count}, Total: {len(wine_data)}")

    print("\n2. Validating data...")
    validate_wine_data(wine_data)
    print("  [PASS] Schema validation")

    print("\n3. Cleaning data...")
    n_before = len(wine_data)
    wine_data = clean_wine_data(wine_data)
    print(f"  Removed {n_before - len(wine_data)} duplicates, {len(wine_data)} samples remaining")

    print("\n4. Saving...")
    wine_data.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")


if __name__ == "__main__":
    main()
