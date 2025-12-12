# Wine Quality Predictor - Makefile

.DEFAULT_GOAL := help

.PHONY: help all clean data analysis report
.PHONY: cl env build run up stop docker-build-push docker-build-local

help: ## Show this help message
	@echo "Wine Quality Predictor - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

all: reports/wine_quality_predictor_report.html ## Run the complete analysis pipeline

clean: ## Remove all generated files
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv
	rm -rf results/models/*.pkl
	rm -rf results/splits.pkl
	rm -rf results/figures/*.png
	rm -rf results/evaluation/*.png
	rm -rf results/evaluation/*.txt
	rm -rf results/evaluation/*.csv
	rm -rf results/evaluation/*.json
	rm -rf reports/wine_quality_predictor_report.html
	rm -rf reports/wine_quality_predictor_report_files/

# Data pipeline
data/raw/winequality-red.csv data/raw/winequality-white.csv: scripts/01_download_data.py
	python scripts/01_download_data.py --output-dir data/raw

data/processed/wine_data_cleaned.csv: scripts/02_clean_data.py \
                                       data/raw/winequality-red.csv \
                                       data/raw/winequality-white.csv
	python scripts/02_clean_data.py \
		--red-wine data/raw/winequality-red.csv \
		--white-wine data/raw/winequality-white.csv \
		--output-path data/processed/wine_data_cleaned.csv

# Analysis pipeline
results/figures/quality_distributions.png results/figures/feature_correlations.png results/figures/correlation_heatmap.png: scripts/03_eda.py \
                                                                                                                              data/processed/wine_data_cleaned.csv
	python scripts/03_eda.py \
		--input-file data/processed/wine_data_cleaned.csv \
		--output-dir results/figures

results/models/rf_wine_models.pkl results/splits.pkl: scripts/04_train_wine_quality_classifier.py \
                                                       data/processed/wine_data_cleaned.csv
	python scripts/04_train_wine_quality_classifier.py \
		--input-csv data/processed/wine_data_cleaned.csv \
		--output-model results/models/rf_wine_models.pkl \
		--output-splits results/splits.pkl

results/evaluation/confusion_matrix_random_forest.png results/evaluation/classification_report.txt: scripts/05_evaluate_using_confusion_matrix.py \
                                                                                                     results/models/rf_wine_models.pkl \
                                                                                                     results/splits.pkl
	python scripts/05_evaluate_using_confusion_matrix.py \
		--model-path results/models/rf_wine_models.pkl \
		--splits-path results/splits.pkl \
		--output-dir results/evaluation

results/evaluation/feature_importance_random_forest.png results/evaluation/feature_importance_table.csv: scripts/06_evaluate_using_feature_importance.py \
                                                                                                          data/processed/wine_data_cleaned.csv \
                                                                                                          results/models/rf_wine_models.pkl
	python scripts/06_evaluate_using_feature_importance.py \
		--input-csv data/processed/wine_data_cleaned.csv \
		--model-path results/models/rf_wine_models.pkl \
		--output-dir results/evaluation

results/models/rf_wine_model_optimized.pkl results/evaluation/rf_hyperparameter_tuning_results.txt results/evaluation/rf_test_metrics.json results/evaluation/confusion_matrix_random_forest_optimized.png: scripts/07_tune_random_forest_hyperparameters.py \
                                                                                                                                                                                                              results/splits.pkl
	python scripts/07_tune_random_forest_hyperparameters.py \
		--splits-pkl results/splits.pkl \
		--output-model results/models/rf_wine_model_optimized.pkl \
		--output-dir results/evaluation \
		--cv-folds 5 \
		--n-jobs -1

# Report
reports/wine_quality_predictor_report.html: reports/wine_quality_predictor_report.qmd \
                                            results/figures/quality_distributions.png \
                                            results/figures/feature_correlations.png \
                                            results/figures/correlation_heatmap.png \
                                            results/evaluation/confusion_matrix_random_forest.png \
                                            results/evaluation/classification_report.txt \
                                            results/evaluation/feature_importance_random_forest.png \
                                            results/evaluation/feature_importance_table.csv \
                                            results/evaluation/confusion_matrix_random_forest_optimized.png \
                                            results/evaluation/rf_test_metrics.json
	quarto render reports/wine_quality_predictor_report.qmd --to html

# Convenience targets
data: data/processed/wine_data_cleaned.csv ## Download and clean data only

analysis: results/evaluation/confusion_matrix_random_forest_optimized.png ## Run analysis without report

report: reports/wine_quality_predictor_report.html ## Render report only

# Environment & Docker
cl: ## Create conda lock for multiple platforms
	conda-lock lock --file environment.yml -p linux-64 -p osx-64 -p osx-arm64 -p win-64 -p linux-aarch64

env: ## Create environment from lock file
	conda env remove -n dockerlock || true
	conda-lock install -n dockerlock conda-lock.yml

build: ## Build docker image
	docker build -t dockerlock --file Dockerfile .

run: up ## Alias for up

up: ## Start docker-compose services
	make stop
	docker-compose up -d

stop: ## Stop docker-compose services
	docker-compose stop

docker-build-push: ## Build and push multi-arch image to Docker Hub
	docker buildx build --platform linux/amd64,linux/arm64 \
		--tag junli73889/docker-condalock-jupyterlab:latest \
		--tag junli73889/docker-condalock-jupyterlab:local-$(shell git rev-parse --short HEAD) \
		--push .

docker-build-local: ## Build single-arch image for local testing
	docker build --tag junli73889/docker-condalock-jupyterlab:local .
