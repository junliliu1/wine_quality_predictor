import subprocess

commands = [
    # 1. Download/Extract Data
    "python scripts/01_download_data.py --output-dir data/raw",

    # 2. Clean/Transform Data
    "python scripts/02_clean_data.py --red-wine data/raw/winequality-red.csv --white-wine data/raw/winequality-white.csv --output-path data/processed/wine_data_cleaned.csv",

    # 3. Exploratory Data Analysis
    "python scripts/03_eda.py --input-file data/processed/cleaned_wine.csv --output-dir results/figures",

    # 4. Model Fitting/Training
    "python scripts/04_train_wine_quality_classifier.py --input-csv data/processed/wine_data_cleaned.csv --output-model results/models/rf_wine_models.pkl",

    # 5a. Model Evaluation – Confusion Matrix
    #"python scripts/05_evaluate_using_confusion_matrix.py --input-csv data/processed/wine_data_cleaned.csv --model-path results/models/rf_wine_models.pkl --output-dir results/evaluation",

    # 5b. Model Evaluation – Feature Importance
    #"python scripts/06_evaluate_using_feature_importance.py --input-csv data/processed/wine_data_cleaned.csv --model-path results/models/rf_wine_models.pkl --output-dir results/evaluation",

    # 5c. Hyperparameter Tuning
    #"python scripts/07_tune_random_forest_hyperparameters.py --input-csv data/processed/wine_data_cleaned.csv --output-model results/models/rf_wine_model_optimized.pkl --output-dir results/evaluation",

    # 6a. Render Report to HTML
    "quarto render reports/wine_quality_predictor_report.qmd --to html",

    # 6b. Render Report to PDF
    "quarto render reports/wine_quality_predictor_report.qmd --to pdf"
]

for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("All steps completed successfully!")
