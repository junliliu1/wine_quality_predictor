# Wine Quality Predictor

## Project Summary

This project predicts wine quality categories (Low, Medium, High) from physicochemical properties using a Random Forest classifier. Using the Wine Quality dataset from the UCI Machine Learning Repository containing 6,497 wine samples (1,599 red and 4,898 white), we analyze 11 chemical properties including alcohol content, acidity levels, and sulfur compounds to classify wine quality. The Random Forest model was chosen for its ability to capture non-linear relationships, provide interpretable feature importance rankings, and handle the class imbalance inherent in quality ratings. Our analysis identifies alcohol content, volatile acidity, and sulphates as the most influential factors in determining wine quality.

## Contributors

The following authors contributed to this project:

* **Junli Liu** ([@junliliu1](https://github.com/junliliu1))
* **Luis Alvarez**
* **Purity Jangaya**
* **Jimmy Wang**

## Dependencies

To ensure reproducibility, this project uses **Conda** for environment management. The key dependencies are listed in `environment.yml`. Major libraries include:

* Python 3.11
* pandas 2.1
* numpy 1.26
* matplotlib 3.8
* seaborn 0.13
* scikit-learn 1.3
* Jupyter / JupyterLab 4.0

*Note: Lock files for different operating systems (macOS Intel, macOS ARM, Linux, Windows) are provided in `conda-lock.yml`.*

## How to Run the Analysis

Follow these steps to set up the environment and run the analysis:

### Step 1: Clone the Repository

```bash
git clone https://github.com/junliliu1/wine_quality_predictor.git
cd wine_quality_predictor
```

### Step 2: Environment Setup

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate wine_quality_predictor
```

### Step 3: Run the Analysis

The data files are already included in the `data/` directory.

1. Launch Jupyter Lab:
```bash
jupyter lab
```

2. Open the analysis file: `wine_quality_predictor_report.ipynb`

3. Run all cells (Kernel > Restart & Run All) to reproduce the analysis and generate the visualizations.

## Project Structure

```
wine_quality_predictor/
├── data/
│   ├── winequality-red.csv       # Red wine dataset (1,599 samples)
│   ├── winequality-white.csv     # White wine dataset (4,898 samples)
│   └── winequality.names         # Dataset documentation
├── wine_quality_predictor_report.ipynb # Main analysis notebook
├── wine_quality_predictor_report.html # Main analysis report
├── environment.yml               # Conda environment specification
├── CODE_OF_CONDUCT.md           # Standards for community behavior
├── CONTRIBUTING.md              # Guidelines for contributing
├── LICENSE.md                   # Project licenses
└── README.md                    # This file
```

## Dataset

**Wine Quality Dataset** (Cortez et al., 2009)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Features: 11 physicochemical properties
- Target: Quality score (3-9), categorized as Low (3-5), Medium (6-7), High (8-9)

## License

This project carries a dual license:

* **Software/Code**: The source code is licensed under the [MIT License](https://opensource.org/licenses/MIT).
* **Report/Analysis**: The narrative report and creative content are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) License.

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553.
