"""
Clean and transform wine quality data.

Usage: python scripts/02_clean_data.py --red-wine data/raw/winequality-red.csv \
           --white-wine data/raw/winequality-white.csv --output-path data/processed/wine_data_cleaned.csv
"""

from pathlib import Path

import click
import pandas as pd
#import pandera as pa
import pandera.pandas as pa
from pandera import Check, Column

EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

FEATURE_RANGES = {
    "fixed acidity": (0, 20),
    "volatile acidity": (0, 2),
    "citric acid": (0, 2),
    "residual sugar": (0, 100),
    "chlorides": (0, 1),
    "free sulfur dioxide": (0, 300),
    "total sulfur dioxide": (0, 500),
    "density": (0.9, 1.1),
    "pH": (2, 5),
    "sulphates": (0, 3),
    "alcohol": (5, 20),
}


def load_and_combine_data(red_path: Path, white_path: Path) -> pd.DataFrame:
    """Load and combine red/white wine datasets."""
    red_wine = pd.read_csv(red_path, sep=";")
    white_wine = pd.read_csv(white_path, sep=";")
    red_wine["wine_type"] = 0
    white_wine["wine_type"] = 1
    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
    print(f"  Red: {len(red_wine)}, White: {len(white_wine)}, Total: {len(wine_data)}")
    return wine_data


def validate_data(df: pd.DataFrame) -> None:
    """Validate data using Pandera."""
    wine_schema = pa.DataFrameSchema(
        {
            "fixed acidity": Column(float, Check.in_range(0, 20), nullable=False),
            "volatile acidity": Column(float, Check.in_range(0, 2), nullable=False),
            "citric acid": Column(float, Check.in_range(0, 2), nullable=False),
            "residual sugar": Column(float, Check.in_range(0, 100), nullable=False),
            "chlorides": Column(float, Check.in_range(0, 1), nullable=False),
            "free sulfur dioxide": Column(
                float, Check.in_range(0, 300), nullable=False
            ),
            "total sulfur dioxide": Column(
                float, Check.in_range(0, 500), nullable=False
            ),
            "density": Column(float, Check.in_range(0.9, 1.1), nullable=False),
            "pH": Column(float, Check.in_range(2, 5), nullable=False),
            "sulphates": Column(float, Check.in_range(0, 3), nullable=False),
            "alcohol": Column(float, Check.in_range(5, 20), nullable=False),
            "quality": Column(int, Check.in_range(3, 9), nullable=False),
        }
    )
    wine_schema.validate(df[EXPECTED_COLUMNS])
    print("  [PASS] Schema validation")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and create quality categories."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {n_before - len(df)} duplicates, {len(df)} samples remaining")

    df["quality_category"] = df["quality"].apply(
        lambda q: "Low (3-5)"
        if q <= 5
        else ("Medium (6-7)" if q <= 7 else "High (8-9)")
    )
    return df


@click.command()
@click.option(
    "--red-wine", type=click.Path(exists=True), required=True, help="Red wine CSV"
)
@click.option(
    "--white-wine", type=click.Path(exists=True), required=True, help="White wine CSV"
)
@click.option(
    "--output-path",
    type=click.Path(),
    default="data/processed/wine_data_cleaned.csv",
    help="Output path",
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

    print("\n2. Validating data...")
    validate_data(wine_data)

    print("\n3. Cleaning data...")
    wine_data = clean_data(wine_data)

    print("\n4. Saving...")
    wine_data.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")


if __name__ == "__main__":
    main()
