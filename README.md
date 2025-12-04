# Wine Quality Predictor

## Project Summary

This project predicts wine quality categories (Low, Medium, High) from physicochemical properties using a Random Forest classifier. Using the Wine Quality dataset from the UCI Machine Learning Repository containing 6,497 wine samples (1,599 red and 4,898 white), we analyze 11 chemical properties including alcohol content, acidity levels, and sulfur compounds to classify wine quality. The Random Forest model was chosen for its ability to capture non-linear relationships, provide interpretable feature importance rankings, and handle the class imbalance inherent in quality ratings. Our analysis identifies alcohol content, volatile acidity, and sulphates as the most influential factors in determining wine quality.

## Contributors

The following authors contributed to this project:

* **Junli Liu** ([@junliliu1](https://github.com/junliliu1))
* **Luis Alvarez** ([@luisalonso8](https://github.com/luisalonso8))
* **Purity Jangaya** ([@PurityJ](https://github.com/Purityj))
* **Jimmy Wang** ([@jimmy2026-V](https://github.com/jimmy2026-V))

## Dependencies

To ensure reproducibility, this project uses **Conda** for environment management. The key dependencies are listed in `environment.yml`. Major libraries include:

* Python 3.11
* pandas 2.1
* numpy 1.26
* matplotlib 3.8
* seaborn 0.13
* scikit-learn 1.3
* Jupyter / JupyterLab 4.0
* Quarto 1.4+
* click 8.0+

*Note: Lock files for different operating systems (macOS Intel, macOS ARM, Linux, Windows) are provided in `conda-lock.yml`.*

## How to Run the Analysis in Jupyter Lab - without Docker

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

The analysis is split into 6 modular Python scripts that should be run in sequence:

```bash
# 1. Download/Extract Data
python scripts/download_data.py \
    --output-dir data/raw

# 2. Clean/Transform Data
python scripts/clean_data.py \
    --red-wine data/raw/winequality-red.csv \
    --white-wine data/raw/winequality-white.csv \
    --output-path data/processed/wine_data_cleaned.csv

# 3. Split and Pre-process Data

# 4. Exploratory Data Analysis

# 5. Model Fitting

# 6. Model Evaluation

# 7. Render the final report
quarto render reports/wine_quality_predictor_report.qmd
```

### Script Details

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `download_data.py` | Download/Extract Data | UCI URLs | `data/raw/*.csv` |
| `clean_data.py` | Clean/Transform Data | Raw CSVs | `data/processed/wine_data_cleaned.csv` |



## How to Run the Analysis in Jupyter Lab using Docker 

[Docker](https://www.docker.com/) is a container solution used to manage the software dependencies for this project. The Docker image used for this project is based on the condaforge/miniforge3:latest image. Additional dependencies are specified in the [Dockerfile](Dockerfile)

## Usage

Follow the instructions below to reproduce the analysis using Docker.

### Setup

1. Installing Docker

If you don't have Docker installed, download and install it from:
- **Windows/Mac**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Follow the [official installation guide](https://docs.docker.com/engine/install/)

To verify Docker is installed correctly, run:
```bash
docker --version
```

2. Clone this GitHub repository and cd to the root of the repository:
 ```
 https://github.com/junliliu1/wine_quality_predictor.git
 cd wine_quality_predictor
 ```

### Running the Analysis

The analysis can be run using either of these options depending on your preference. Run all these commands on a terminal from the root of the project.


#### Option 1: Build the Docker image locally (recommended for active development)

```
docker-compose up --build
```

This will build the Docker image and start the container.
Once the container is running, a link to access JupyterLab will be shown in the terminal. Look for a URL that starts with http://127.0.0.1:8888/lab?. Copy and paste that URL into your browser.

Changes made in notebook are reflected locally in cloned repo. Commit and push changes to github for others to access.
This is best if you want to actively work on the project and potentially rebuild the environment.


#### Clean Up

To shut down the container and clean up the resources, type `Ctrl` + `C` in the terminal where you launched the container, and then type:

```bash
docker compose rm
```

#### Option 2: Pull a prebuilt image from DockerHub (faster, reproducible environment)

1. Pull the image from Dockerhub

```bash
docker pull junli73889/wine-quality-predictor:latest
```

2. Run the container

```bash
docker run -it -p 8888:8888 -v $(pwd):/workplace junli73889/wine-quality-predictor:latest
```

3. The container will start and provide a localhost URL. Look for a URL that starts with http://127.0.0.1:8888/lab?. Copy and paste that URL into your browser.


#### Clean Up

To shut down the container and clean up the resources, type `Ctrl` + `C` in the terminal where you launched the container, and then:

- If it’s running in the background, then: `docker ps`
- Find the container name or ID, then: : `docker stop <container_id>`
- Stopping it does NOT delete it. To remove it: `docker rm <container_id>`


## Adding a new dependency

1. Add the dependency to the Dockerfile file on a new branch.
2. Re-build the Docker image locally to ensure it builds and runs properly.
3. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.
4. Update the docker-compose.yml file on your branch to use the new container image (make sure to update the tag specifically).
5. Send a pull request to merge the changes into the main branch.


## Project Structure

```
TODO
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
