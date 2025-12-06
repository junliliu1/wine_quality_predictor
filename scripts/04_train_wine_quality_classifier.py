"""

This script trains a Random Forest classifier to predict wine quality
from physicochemical properties.

his script trains a Random Forest classifier to predict wine quality
from physicochemical properties. 

It performs the following steps:
1. Loads processed wine data from a CSV file.
2. Splits the data into features and target labels.
3. Splits data into training and test sets.
4. Trains a Random Forest classifier.
5. Evaluates model performance (training accuracy, test accuracy, OOB score, cross-validation).
6. Saves the trained model to a specified file path.

Usage (from terminal):
$ python train_wine_quality_classifier.py --input-csv data/processed/wine_data_cleaned.csv --output-model models/rf_wine_models.pkl

"""
from pathlib import Path
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

@click.command()
@click.option( "--input-csv", type=click.Path(exists=True), required=True, help="Path to the processed wine data CSV")
@click.option("--output-model", type=click.Path(), default="models/rf_wine_models.pkl", help="Path to save the trained model")
@click.option("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
@click.option("--random-state", type=int, default=42, help="Random seed for reproducibility")
def main(input_csv: str, output_model: str, test_size: float, random_state: int):
    """Train the Random Forest classifier on wine quality data."""
    print("=" * 60)
    print("STEP 4: TRAIN MODEL")
    print("=" * 60)

    # 1. Load processed data
    data = pd.read_csv(input_csv)
    print(f"Loaded {len(data)} samples from {input_csv}")

    # 2. SPlit into features (X) and (y)
    X = data.drop(columns=["quality", "quality_category"]) 
    y = data["quality"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

    # 4. Build Random Forest model
    print("\nBuilding Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, oob_score=True, n_jobs=1)

    # 5. Train the model
    rf_model.fit(X_train, y_train)

    #6. Predictions and performance
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    oob_accuracy = rf_model.oob_score_

    print("\nRandom Forest Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"OOB Score: {oob_accuracy:.4f}")

    # 7. Cross-validation 
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=4, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 8. Save trained model
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"\nSaved trained model to {output_model}")

if __name__ =="__main__":
    main()