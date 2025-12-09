"""
This script trains a Random Forest classifier to predict wine quality
from physicochemical properties.

Steps:
1. Load processed wine data from a CSV file.
2. Split data into features and target labels.
3. Split data into training and test sets (stratified).
4. Train a Random Forest classifier.
5. Evaluate model performance (training/test accuracy, OOB, cross-validation).
6. Save the trained model and train/test splits for reproducibility.

Usage (from terminal):
$ scripts/04_train_wine_quality_classifier.py \
    --input-csv data/processed/wine_data_cleaned.csv \
    --output-model results/models/rf_wine_models.pkl \
    --output-splits results/splits.pkl
"""

from pathlib import Path
import pickle

import click
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_data(input_csv: str):
    data = pd.read_csv(input_csv)
    X = data.drop(columns=["quality", "quality_category"])
    y = data["quality_category"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, le

def train_model(X_train, y_train, random_state=42):
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=random_state, oob_score=True, n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    return rf_model


@click.command()
@click.option("--input-csv", type=click.Path(exists=True), required=True,
              help="Path to processed wine CSV file")
@click.option("--output-model", type=click.Path(), default="results/models/rf_wine_models.pkl",
              help="Path to save the trained model")
@click.option("--output-splits", type=click.Path(), default="results/splits.pkl",
              help="Path to save train/test splits for reproducibility")
@click.option("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
@click.option("--random-state", type=int, default=42, help="Random seed for reproducibility")
def main(input_csv, output_model, output_splits, test_size, random_state):
    print("="*60)
    print("STEP 4: TRAIN RANDOM FOREST MODEL")
    print("="*60)

    # 1. Load data
    X, y = load_data(input_csv)
    print(f"Loaded {len(X)} samples from {input_csv}")

    # 2. Split data
    X_train, X_test, y_train, y_test, le = split_data(X, y, test_size=test_size, random_state=random_state)
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

    # Show class distributions
    print("\nTraining set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, label in enumerate(le.classes_):
        print(f"  {label}: {counts[i]} ({counts[i]/len(y_train)*100:.1f}%)")

    print("\nTest set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, label in enumerate(le.classes_):
        print(f"  {label}: {counts[i]} ({counts[i]/len(y_test)*100:.1f}%)")

    # 3. Train model
    print("\nTraining Random Forest...")
    rf_model = train_model(X_train, y_train, random_state=random_state)

    # 4. Performance metrics
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    oob_score = rf_model.oob_score_

    print("\nRandom Forest Performance:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"OOB Score: {oob_score:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=4, scoring="accuracy")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 5. Save model
    model_path = Path(output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"\nSaved trained model to {model_path}")

    # 6. Save splits
    splits_path = Path(output_splits)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test, le), f)
    print(f"Saved train/test splits to {splits_path}")

if __name__ == "__main__":
    main()
