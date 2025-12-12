"""
Exploratory Data Analysis (EDA) Module

This module provides reusable functions for performing exploratory data analysis
on wine quality datasets, including visualizations for quality distributions,
feature correlations, and correlation heatmaps.

Functions:
    create_quality_distributions: Generate quality distribution plots
    create_feature_correlations: Create feature correlation bar chart
    create_correlation_heatmap: Generate correlation heatmap for features
    perform_eda: Orchestrate complete EDA workflow
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .data_cleaning import categorize_quality


def create_quality_distributions(
    wine_data: pd.DataFrame,
    output_dir: str | Path
) -> pd.DataFrame:
    """
    Create quality distribution visualizations.

    Generates three subplots:
    1. Overall wine quality distribution
    2. Quality distribution by wine type (red/white)
    3. Quality categories pie chart

    Parameters
    ----------
    wine_data : pd.DataFrame
        Wine dataset with 'quality', 'wine_type', and 'quality_category' columns
    output_dir : str or Path
        Directory to save the visualization

    Returns
    -------
    pd.DataFrame
        The input wine_data (unchanged)

    Raises
    ------
    ValueError
        If required columns are missing from wine_data
    """
    required_cols = ['quality', 'wine_type', 'quality_category']
    missing_cols = [col for col in required_cols if col not in wine_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overall quality distribution
    wine_data['quality'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color='steelblue'
    )
    axes[0].set_title('Overall Wine Quality Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Quality by wine type
    quality_by_type = wine_data.groupby(['wine_type', 'quality']).size().unstack(fill_value=0)
    quality_by_type.T.plot(kind='bar', ax=axes[1], color=['darkred', 'gold'], alpha=0.8)
    axes[1].set_title('Quality Distribution by Wine Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Quality Score')
    axes[1].set_ylabel('Count')
    axes[1].legend(['Red Wine', 'White Wine'])
    axes[1].grid(True, alpha=0.3)

    # Quality categories pie chart
    category_counts = wine_data['quality_category'].value_counts()
    category_counts.plot(
        kind='pie',
        ax=axes[2],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff', '#99ff99']
    )
    axes[2].set_title('Quality Categories for Classification', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_path / 'quality_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Figure 1. Quality distributions saved")
    return wine_data


def create_feature_correlations(
    wine_data: pd.DataFrame,
    output_dir: str | Path
) -> pd.Index:
    """
    Create feature correlation bar chart.

    Generates a horizontal bar chart showing correlations between
    physicochemical features and wine quality.

    Parameters
    ----------
    wine_data : pd.DataFrame
        Wine dataset with numeric features and 'quality' column
    output_dir : str or Path
        Directory to save the visualization

    Returns
    -------
    pd.Index
        Index of feature columns used in correlation analysis

    Raises
    ------
    ValueError
        If 'quality' column is missing from wine_data
    """
    if 'quality' not in wine_data.columns:
        raise ValueError("wine_data must contain 'quality' column")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get only numeric columns, excluding quality and quality_category
    feature_cols = wine_data.select_dtypes(include=[np.number]).columns.drop(['quality'])
    correlations = wine_data[feature_cols].corrwith(wine_data['quality']).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in correlations.values]
    correlations.plot(kind='barh', color=colors)
    plt.title('Feature Correlations with Wine Quality', fontsize=14, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Figure 2. Feature correlations saved")
    print("\nTop 5 Positive Correlations:")
    print(correlations.head())
    print("\nTop 5 Negative Correlations:")
    print(correlations.tail())

    return feature_cols


def create_correlation_heatmap(
    wine_data: pd.DataFrame,
    feature_cols: pd.Index,
    output_dir: str | Path
) -> None:
    """
    Create correlation heatmap for physicochemical features.

    Generates a triangular heatmap showing pairwise correlations
    between all numeric features.

    Parameters
    ----------
    wine_data : pd.DataFrame
        Wine dataset with numeric features
    feature_cols : pd.Index
        Column names to include in the heatmap
    output_dir : str or Path
        Directory to save the visualization

    Raises
    ------
    ValueError
        If any feature_cols are missing from wine_data
    """
    missing_cols = [col for col in feature_cols if col not in wine_data.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    correlation_matrix = wine_data[feature_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Figure 3. Correlation heatmap saved")


def perform_eda(
    input_file: str | Path,
    output_dir: str | Path
) -> None:
    """
    Perform complete exploratory data analysis workflow.

    Orchestrates the entire EDA process:
    1. Loads cleaned wine data
    2. Adds quality categories
    3. Creates all visualizations (distributions, correlations, heatmap)

    Parameters
    ----------
    input_file : str or Path
        Path to cleaned wine CSV file
    output_dir : str or Path
        Directory to save all EDA outputs

    Raises
    ------
    FileNotFoundError
        If input_file does not exist
    ValueError
        If input data is invalid or missing required columns
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WINE QUALITY EDA")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {input_file}")
    wine_data = pd.read_csv(input_file)
    print(f"Dataset shape: {wine_data.shape[0]} rows, {wine_data.shape[1]} columns")

    # Add quality categories
    wine_data['quality_category'] = wine_data['quality'].apply(categorize_quality)

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    print()

    wine_data = create_quality_distributions(wine_data, output_dir)
    print()
    feature_cols = create_feature_correlations(wine_data, output_dir)
    print()
    create_correlation_heatmap(wine_data, feature_cols, output_dir)

    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print("=" * 60)
    print(f"\nAll plots saved to: {output_path}")
