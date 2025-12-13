"""
Perform simple tests for the EDA module.
"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eda import categorize_quality, create_quality_distributions


class TestCategorizeQuality:
    """Simple test for categorize_quality function."""

    def test_categorize_quality_basic(self):
        """Test that quality scores are categorized correctly."""
        assert categorize_quality(3) == "Low (3-5)"
        assert categorize_quality(6) == "Medium (6-7)"
        assert categorize_quality(8) == "High (8-9)"


class TestCreateQualityDistributions:
    """Simple test for create_quality_distributions function."""

    def test_creates_plot_file(self, tmp_path):
        """Test that the function creates a plot file."""
        # Create simple test data
        test_data = pd.DataFrame({
            'quality': [5, 6, 7, 8],
            'wine_type': [0, 0, 1, 1],
            'quality_category': ['Low (3-5)', 'Medium (6-7)', 'Medium (6-7)', 'High (8-9)']
        })
        
        # Run function
        result = create_quality_distributions(test_data, tmp_path)
        
        # Check that file was created
        output_file = tmp_path / 'quality_distributions.png'
        assert output_file.exists()
        
        # Check that data is returned unchanged
        assert len(result) == 4
        assert 'quality' in result.columns