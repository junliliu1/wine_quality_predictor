"""
Clean and combine wine quality datasets

Usage: python scripts/cleaning.py --input-dir data/raw --output-dir data/processed
"""

import click
import pandas as pd
from pathlib import Path


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), default="data/raw",
              help="Directory containing raw wine data")
@click.option("--output-dir", type=click.Path(), default="data/processed",
              help="Output directory for cleaned data")
def main(input_dir, output_dir):
    """Clean and combine red and white wine datasets."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 2: CLEAN AND COMBINE DATA")
    print("=" * 60)
    
    # Load both datasets
    print("\nLoading datasets...")
    red_wine = pd.read_csv(input_path / "winequality-red.csv", sep=';')
    white_wine = pd.read_csv(input_path / "winequality-white.csv", sep=';')
    print(f"  Red wine: {red_wine.shape[0]} rows")
    print(f"  White wine: {white_wine.shape[0]} rows")
    
    # Add wine type column
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    # Combine both datasets
    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
    print(f"\nCombined dataset: {wine_data.shape[0]} rows, {wine_data.shape[1]} columns")
    
    # Remove all duplicates
    original_size = len(wine_data)
    wine_data = wine_data.drop_duplicates()
    print(f"Removed {original_size - len(wine_data)} duplicates")
    
    # Check for missing values
    missing = wine_data.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    # Save cleaned data
    output_file = output_path / "cleaned_wine.csv"
    wine_data.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")
    print(f"Final shape: {wine_data.shape[0]} rows, {wine_data.shape[1]} columns")


if __name__ == "__main__":
    main()