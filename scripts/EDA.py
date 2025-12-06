"""
Wine Quality EDA Script

Performs exploratory data analysis on cleaned wine quality dataset,
generating visualizations for quality distributions, feature correlations,
and correlation heatmaps.

Usage: 
    python scripts/eda.py --input-file data/processed/cleaned_wine.csv --output-dir results/eda

Arguments:
    --input-file: Path to cleaned wine data CSV (default: data/processed/cleaned_wine.csv)
    --output-dir: Output directory for plots (default: results/eda)

Outputs:
    - quality_distributions.png
    - feature_correlations.png
    - correlation_heatmap.png
"""

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def categorize_quality(quality):
    """Categorize quality into Low, Medium, High."""
    if quality <= 5:
        return 'Low (3-5)'
    elif quality <= 7:
        return 'Medium (6-7)'
    else:
        return 'High (8-9)'


@click.command()
@click.option("--input-file", type=click.Path(exists=True), default="data/processed/cleaned_wine.csv",
              help="Path to cleaned wine data CSV")
@click.option("--output-dir", type=click.Path(), default="results/eda",
              help="Output directory for EDA plots and results")
def main(input_file, output_dir):
    """Perform EDA on cleaned wine quality data."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 3: EDA")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    wine_data = pd.read_csv(input_file)
    wine_data['quality_category'] = wine_data['quality'].apply(categorize_quality)
    print(f"Dataset shape: {wine_data.shape[0]} rows, {wine_data.shape[1]} columns")
    
    # Figure 1: Quality distributions
    print("\nCreating quality distribution plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    wine_data['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Overall Wine Quality Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    quality_by_type = wine_data.groupby(['wine_type', 'quality']).size().unstack(fill_value=0)
    quality_by_type.T.plot(kind='bar', ax=axes[1], color=['darkred', 'gold'], alpha=0.8)
    axes[1].set_title('Quality Distribution by Wine Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Quality Score')
    axes[1].set_ylabel('Count')
    axes[1].legend(['Red Wine', 'White Wine'])
    axes[1].grid(True, alpha=0.3)
    
    category_counts = wine_data['quality_category'].value_counts()
    category_counts.plot(kind='pie', ax=axes[2], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[2].set_title('Quality Categories for Classification', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path / 'quality_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path / 'quality_distributions.png'}")
    plt.close()
    
    # Figure 2: Feature correlations
    print("Creating feature correlation plot...")
    feature_cols = wine_data.columns.drop(['quality', 'quality_category', 'wine_type'])
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
    print(f"  Saved: {output_path / 'feature_correlations.png'}")
    plt.close()
    
    # Figure 3: Correlation heatmap
    print("Creating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = wine_data[feature_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path / 'correlation_heatmap.png'}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print("=" * 60)
    print(f"All plots saved to: {output_path}")


if __name__ == "__main__":
    main()