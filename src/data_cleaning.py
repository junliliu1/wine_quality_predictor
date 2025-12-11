"""
Data cleaning utilities for wine quality prediction.

This module provides functions to load, validate, and clean wine quality datasets.

Functions:
    load_and_combine_data: Load and merge red/white wine datasets.
    validate_wine_data: Validate data against expected schema using Pandera.
    categorize_quality: Categorize wine quality into Low/Medium/High.
    clean_wine_data: Remove duplicates and add quality categories.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column


# Expected columns in the wine dataset
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

# Valid ranges for each feature
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


def load_and_combine_data(
    red_path: Union[str, Path], white_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load and combine red and white wine datasets.

    Reads both CSV files (semicolon-separated), adds a 'wine_type' column
    (0 for red, 1 for white), and concatenates them into a single DataFrame.

    Parameters
    ----------
    red_path : str or Path
        Path to the red wine CSV file.
    white_path : str or Path
        Path to the white wine CSV file.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all wine samples and a 'wine_type' column.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    pd.errors.EmptyDataError
        If either file is empty.

    Examples
    --------
    >>> df = load_and_combine_data("data/red.csv", "data/white.csv")
    >>> df.shape
    (6497, 13)
    """
    red_path = Path(red_path)
    white_path = Path(white_path)

    if not red_path.exists():
        raise FileNotFoundError(f"Red wine file not found: {red_path}")
    if not white_path.exists():
        raise FileNotFoundError(f"White wine file not found: {white_path}")

    red_wine = pd.read_csv(red_path, sep=";")
    white_wine = pd.read_csv(white_path, sep=";")

    red_wine["wine_type"] = 0
    white_wine["wine_type"] = 1

    combined = pd.concat([red_wine, white_wine], ignore_index=True)
    return combined


def validate_wine_data(df: pd.DataFrame) -> bool:
    """
    Validate wine data against expected schema using Pandera.

    Checks that all expected columns exist and values are within valid ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    pa.errors.SchemaError
        If validation fails (missing columns, out-of-range values, etc.).
    ValueError
        If DataFrame is empty.

    Examples
    --------
    >>> validate_wine_data(wine_df)
    True
    """
    if df.empty:
        raise ValueError("Cannot validate empty DataFrame")

    wine_schema = pa.DataFrameSchema(
        {
            "fixed acidity": Column(float, Check.in_range(0, 20), nullable=False),
            "volatile acidity": Column(float, Check.in_range(0, 2), nullable=False),
            "citric acid": Column(float, Check.in_range(0, 2), nullable=False),
            "residual sugar": Column(float, Check.in_range(0, 100), nullable=False),
            "chlorides": Column(float, Check.in_range(0, 1), nullable=False),
            "free sulfur dioxide": Column(float, Check.in_range(0, 300), nullable=False),
            "total sulfur dioxide": Column(float, Check.in_range(0, 500), nullable=False),
            "density": Column(float, Check.in_range(0.9, 1.1), nullable=False),
            "pH": Column(float, Check.in_range(2, 5), nullable=False),
            "sulphates": Column(float, Check.in_range(0, 3), nullable=False),
            "alcohol": Column(float, Check.in_range(5, 20), nullable=False),
            "quality": Column(int, Check.in_range(3, 9), nullable=False),
        }
    )
    wine_schema.validate(df[EXPECTED_COLUMNS])
    return True


def categorize_quality(quality: int) -> str:
    """
    Categorize wine quality score into descriptive categories.

    Maps numeric quality scores (3-9) to three categories:
    - Low (3-5): Lower quality wines
    - Medium (6-7): Average quality wines
    - High (8-9): Premium quality wines

    Parameters
    ----------
    quality : int
        Wine quality score (expected range: 3-9).

    Returns
    -------
    str
        Quality category: "Low (3-5)", "Medium (6-7)", or "High (8-9)".

    Examples
    --------
    >>> categorize_quality(4)
    'Low (3-5)'
    >>> categorize_quality(6)
    'Medium (6-7)'
    >>> categorize_quality(8)
    'High (8-9)'
    """
    if quality <= 5:
        return "Low (3-5)"
    elif quality <= 7:
        return "Medium (6-7)"
    else:
        return "High (8-9)"


def clean_wine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean wine data by removing duplicates and adding quality categories.

    Performs the following cleaning steps:
    1. Remove duplicate rows
    2. Reset index
    3. Add 'quality_category' column based on quality score

    Parameters
    ----------
    df : pd.DataFrame
        Raw wine DataFrame with 'quality' column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with duplicates removed and 'quality_category' added.

    Raises
    ------
    KeyError
        If 'quality' column is missing from the DataFrame.

    Examples
    --------
    >>> cleaned_df = clean_wine_data(raw_df)
    >>> 'quality_category' in cleaned_df.columns
    True
    """
    if "quality" not in df.columns:
        raise KeyError("DataFrame must contain 'quality' column")

    df = df.drop_duplicates().reset_index(drop=True)
    df["quality_category"] = df["quality"].apply(categorize_quality)
    return df
