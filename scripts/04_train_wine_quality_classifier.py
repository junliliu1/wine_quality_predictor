"""
scripts/04_train_wine_quality_classifier.py

This script serves as the command-line interface (CLI) to train a Random Forest 
classifier on the cleaned wine quality dataset. It uses the reusable functions 
defined in src/train_wine_classifier.py.

Steps performed:
1. Load the processed wine dataset
2. Split data into training and test sets
3. Train a Random Forest model
4. Evaluate model performance and print metrics
5. Save the trained model and train/test splits

Usage (from terminal):
$ python scripts/04_train_wine_quality_classifier.py \
    --input-csv data/processed/wine_data_cleaned.csv \
    --output-model results/models/rf_wine_models.pkl \
    --output-splits results/splits.pkl
"""

import click
from pathlib import Path
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.train_wine_quality_classifier import (
    load_data, split_data, train_model, evaluate_model,
    save_model, save_splits, ensure_dir_exists
)

@click.command()
@click.option("--input-csv", type=click.Path(exists=True), required=True)
@click.option("--output-model", type=click.Path(), default="results/models/rf_wine_models.pkl")
@click.option("--output-splits", type=click.Path(), default="results/splits.pkl")
@click.option("--test-size", type=float, default=0.2)
@click.option("--random-state", type=int, default=42)
def main(input_csv, output_model, output_splits, test_size, random_state):
    print("="*60)
    print("STEP 4: TRAIN RANDOM FOREST MODEL")
    print("="*60)

    ensure_dir_exists(Path(output_model).parent)
    ensure_dir_exists(Path(output_splits).parent)

    # Load data
    X, y = load_data(input_csv)
    print(f"Loaded {len(X)} samples from {input_csv}")

    # Split data
    X_train, X_test, y_train, y_test, le = split_data(X, y, test_size, random_state)
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples\n")

    # Show class distributions
    print("Training set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, label in enumerate(le.classes_):
        print(f"  {label}: {counts[i]} ({counts[i]/len(y_train)*100:.1f}%)")

    print("\nTest set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, label in enumerate(le.classes_):
        print(f"  {label}: {counts[i]} ({counts[i]/len(y_test)*100:.1f}%)")

    # Train model
    print("\nTraining Random Forest...")
    rf_model = train_model(X_train, y_train, random_state)

    # Evaluate model
    metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    print("\nRandom Forest Performance:")
    print(f"Training Accuracy: {metrics['train_acc']:.4f}")
    print(f"Test Accuracy: {metrics['test_acc']:.4f}")
    print(f"OOB Score: {metrics['oob_score']:.4f}")
    print(f"Cross-validation Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

    # Save outputs
    save_model(rf_model, output_model)
    print(f"\nSaved trained model to {output_model}")
    save_splits((X_train, X_test, y_train, y_test, le), output_splits)
    print(f"Saved train/test splits to {output_splits}")


if __name__ == "__main__":
    main()
