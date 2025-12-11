"""
Tests for the data_cleaning module.

This module tests the data cleaning functions including loading, validation,
categorization, and cleaning of wine quality data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import (
    load_and_combine_data,
    validate_wine_data,
    categorize_quality,
    clean_wine_data,
    EXPECTED_COLUMNS,
)


class TestLoadAndCombineData:
    """Tests for the load_and_combine_data function."""

    @pytest.fixture
    def sample_wine_data(self, tmp_path):
        """Create sample red and white wine CSV files."""
        red_data = """fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality
7.4;0.70;0.00;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
7.8;0.88;0.00;2.6;0.098;25;67;0.9968;3.20;0.68;9.8;5"""

        white_data = """fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality
7.0;0.27;0.36;20.7;0.045;45;170;1.0010;3.00;0.45;8.8;6
6.3;0.30;0.34;1.6;0.049;14;132;0.9940;3.30;0.49;9.5;6"""

        red_path = tmp_path / "red.csv"
        white_path = tmp_path / "white.csv"
        red_path.write_text(red_data)
        white_path.write_text(white_data)

        return red_path, white_path

    def test_load_and_combine_basic(self, sample_wine_data):
        """Test basic loading and combining of wine datasets."""
        red_path, white_path = sample_wine_data
        result = load_and_combine_data(red_path, white_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 2 red + 2 white
        assert "wine_type" in result.columns

    def test_load_and_combine_wine_type_values(self, sample_wine_data):
        """Test that wine_type is correctly assigned (0=red, 1=white)."""
        red_path, white_path = sample_wine_data
        result = load_and_combine_data(red_path, white_path)

        # First 2 rows should be red (0), last 2 should be white (1)
        assert result.iloc[0]["wine_type"] == 0
        assert result.iloc[1]["wine_type"] == 0
        assert result.iloc[2]["wine_type"] == 1
        assert result.iloc[3]["wine_type"] == 1

    def test_load_and_combine_all_columns_present(self, sample_wine_data):
        """Test that all expected columns are present after combining."""
        red_path, white_path = sample_wine_data
        result = load_and_combine_data(red_path, white_path)

        for col in EXPECTED_COLUMNS:
            assert col in result.columns
        assert "wine_type" in result.columns

    def test_load_and_combine_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        red_path = tmp_path / "nonexistent_red.csv"
        white_path = tmp_path / "nonexistent_white.csv"

        with pytest.raises(FileNotFoundError):
            load_and_combine_data(red_path, white_path)

    def test_load_and_combine_string_paths(self, sample_wine_data):
        """Test that string paths are accepted."""
        red_path, white_path = sample_wine_data
        result = load_and_combine_data(str(red_path), str(white_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4


class TestValidateWineData:
    """Tests for the validate_wine_data function."""

    @pytest.fixture
    def valid_wine_df(self):
        """Create a valid wine DataFrame for testing."""
        return pd.DataFrame({
            "fixed acidity": [7.4, 7.8],
            "volatile acidity": [0.70, 0.88],
            "citric acid": [0.00, 0.00],
            "residual sugar": [1.9, 2.6],
            "chlorides": [0.076, 0.098],
            "free sulfur dioxide": [11.0, 25.0],
            "total sulfur dioxide": [34.0, 67.0],
            "density": [0.9978, 0.9968],
            "pH": [3.51, 3.20],
            "sulphates": [0.56, 0.68],
            "alcohol": [9.4, 9.8],
            "quality": [5, 5],
        })

    def test_validate_wine_data_valid(self, valid_wine_df):
        """Test that valid data passes validation."""
        assert validate_wine_data(valid_wine_df) is True

    def test_validate_wine_data_empty_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot validate empty DataFrame"):
            validate_wine_data(empty_df)

    def test_validate_wine_data_out_of_range_alcohol(self, valid_wine_df):
        """Test that out-of-range alcohol values fail validation."""
        invalid_df = valid_wine_df.copy()
        invalid_df.loc[0, "alcohol"] = 25.0  # Out of range (max is 20)

        with pytest.raises(Exception):  # Pandera SchemaError
            validate_wine_data(invalid_df)

    def test_validate_wine_data_out_of_range_ph(self, valid_wine_df):
        """Test that out-of-range pH values fail validation."""
        invalid_df = valid_wine_df.copy()
        invalid_df.loc[0, "pH"] = 1.0  # Out of range (min is 2)

        with pytest.raises(Exception):  # Pandera SchemaError
            validate_wine_data(invalid_df)

    def test_validate_wine_data_out_of_range_density(self, valid_wine_df):
        """Test that out-of-range density values fail validation."""
        invalid_df = valid_wine_df.copy()
        invalid_df.loc[0, "density"] = 0.5  # Out of range (min is 0.9)

        with pytest.raises(Exception):  # Pandera SchemaError
            validate_wine_data(invalid_df)


class TestCategorizeQuality:
    """Tests for the categorize_quality function."""

    def test_categorize_quality_low(self):
        """Test that scores 3-5 are categorized as Low."""
        assert categorize_quality(3) == "Low (3-5)"
        assert categorize_quality(4) == "Low (3-5)"
        assert categorize_quality(5) == "Low (3-5)"

    def test_categorize_quality_medium(self):
        """Test that scores 6-7 are categorized as Medium."""
        assert categorize_quality(6) == "Medium (6-7)"
        assert categorize_quality(7) == "Medium (6-7)"

    def test_categorize_quality_high(self):
        """Test that scores 8-9 are categorized as High."""
        assert categorize_quality(8) == "High (8-9)"
        assert categorize_quality(9) == "High (8-9)"

    def test_categorize_quality_boundary_values(self):
        """Test boundary values between categories."""
        assert categorize_quality(5) == "Low (3-5)"
        assert categorize_quality(6) == "Medium (6-7)"
        assert categorize_quality(7) == "Medium (6-7)"
        assert categorize_quality(8) == "High (8-9)"


class TestCleanWineData:
    """Tests for the clean_wine_data function."""

    @pytest.fixture
    def wine_df_with_duplicates(self):
        """Create a wine DataFrame with duplicate rows."""
        return pd.DataFrame({
            "fixed acidity": [7.4, 7.4, 7.8],
            "volatile acidity": [0.70, 0.70, 0.88],
            "quality": [5, 5, 6],
        })

    def test_clean_wine_data_removes_duplicates(self, wine_df_with_duplicates):
        """Test that duplicate rows are removed."""
        result = clean_wine_data(wine_df_with_duplicates)
        assert len(result) == 2  # 3 rows - 1 duplicate = 2 unique

    def test_clean_wine_data_adds_quality_category(self, wine_df_with_duplicates):
        """Test that quality_category column is added."""
        result = clean_wine_data(wine_df_with_duplicates)
        assert "quality_category" in result.columns

    def test_clean_wine_data_correct_categories(self):
        """Test that quality categories are correctly assigned."""
        df = pd.DataFrame({
            "quality": [3, 5, 6, 7, 8, 9],
            "other_col": [1, 2, 3, 4, 5, 6],
        })
        result = clean_wine_data(df)

        assert result.loc[0, "quality_category"] == "Low (3-5)"
        assert result.loc[1, "quality_category"] == "Low (3-5)"
        assert result.loc[2, "quality_category"] == "Medium (6-7)"
        assert result.loc[3, "quality_category"] == "Medium (6-7)"
        assert result.loc[4, "quality_category"] == "High (8-9)"
        assert result.loc[5, "quality_category"] == "High (8-9)"

    def test_clean_wine_data_resets_index(self, wine_df_with_duplicates):
        """Test that index is reset after removing duplicates."""
        result = clean_wine_data(wine_df_with_duplicates)
        assert list(result.index) == [0, 1]

    def test_clean_wine_data_missing_quality_column(self):
        """Test that KeyError is raised when quality column is missing."""
        df = pd.DataFrame({
            "fixed acidity": [7.4, 7.8],
            "other_col": [1, 2],
        })
        with pytest.raises(KeyError, match="quality"):
            clean_wine_data(df)

    def test_clean_wine_data_preserves_other_columns(self):
        """Test that other columns are preserved after cleaning."""
        df = pd.DataFrame({
            "quality": [5, 6],
            "feature1": [1.0, 2.0],
            "feature2": ["a", "b"],
        })
        result = clean_wine_data(df)

        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert list(result["feature1"]) == [1.0, 2.0]
