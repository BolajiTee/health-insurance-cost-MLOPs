Health Insurance Cost — MLOps Project

TL;DR
This repository contains an end-to-end machine learning project for predicting health insurance costs. It includes data preprocessing, model training (CatBoost), a prediction pipeline, a small application for inference, and MLOps artifacts (logs, training metadata).

Table of Contents
- Project Overview
- Quickstart
- Repository Structure
- Setup & Installation
- Data
- Usage
	- Training
	- Generating predictions
	- Running the app
- Model & MLOps notes
- Tests & Validation
- Contributing
- License

Project Overview
This project demonstrates a production-oriented workflow for training and serving a model that predicts health insurance costs. It includes:
- Data ingestion and preprocessing pipelines
- Model training with CatBoost saved artifacts
- A prediction pipeline for batch and single-record inference
- Simple application entrypoints for demo and local testing
- Logs and training metadata for reproducibility

Quickstart
1. Create and activate a virtual environment (recommended Python 3.8+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training (example):

```bash
python model_trainer.py
```

4. Run the prediction pipeline or app:

```bash
python prediction_pipeline.py
# or
python app.py
```

Repository Structure
- app.py — small application / API entrypoint for inference and demo.
- model_trainer.py — script to train the model, log metrics, and save artifacts.
- prediction_pipeline.py — preprocessing + inference pipeline for producing predictions.
- load_transformation.py — helpers to load/save preprocessing objects and transform incoming data.
- utils.py — shared utilities and helper functions used across scripts.
- log_exception.py — logging and exception helpers.
- requirements.txt — Python dependencies.
- data/ — raw and small sample CSVs used for experiments.
- datasets/ — processed train/test/prediction CSVs and outputs (predictions.csv).
- logs/ — training and runtime logs.
- catboost_info/ — CatBoost training metadata and artifacts.
- static/, templates/ — assets and templates used by the app (if serving HTML pages).

Setup & Installation
1. Clone the repository and change directory:

```bash
git clone <repo-url>
cd health-insurance-cost-MLOPs
```

2. Create & activate a virtual environment (examples):

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

UNIX / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data
- The repository includes example CSVs in `data/` and `datasets/`. Replace these with your production data as needed.
- Keep any sensitive data out of the repo. Use environment variables or a secret manager to provide credentials and private datasets.

Usage

Training
- `model_trainer.py` contains the training logic. It reads processed data, trains a CatBoost model, saves model artifacts under `catboost_info/` (or another configured output), and writes logs to `logs/`.
- Example:

```bash
python model_trainer.py --data datasets/train_data.csv --output catboost_info/
```

Generating predictions
- `prediction_pipeline.py` loads the saved preprocessing and model artifacts and produces predictions for a given CSV or single record.
- Example (batch):

```bash
python prediction_pipeline.py --input datasets/test_data.csv --output datasets/predictions.csv
```

Running the app
- `app.py` provides a simple demo/web UI or API to run inference interactively. Start it locally for manual testing:

```bash
python app.py
# visit http://localhost:5000 (or the printed address)
```

Model & MLOps notes
- Model: CatBoost is used (see `requirements.txt` and `catboost_info/`).
- Artifacts: Keep model binary, preprocessing objects, and training metadata together for reproducibility.
- Logging: Training logs are stored in `logs/` and CatBoost also writes training metadata in `catboost_info/`.
- Reproducibility: Pin dependency versions in `requirements.txt` and save random seeds and model parameters alongside artifacts.

Tests & Validation
- This repo currently includes a notebook (`notebook/health_insurance_cost_mlops.ipynb`) for exploratory analysis and manual validation. Adding automated unit tests and CI is recommended.
- Suggested tests:
	- Unit tests for preprocessing functions in `load_transformation.py` and `utils.py`.
	- Integration test for `prediction_pipeline.py` using a small sample CSV.

Contributing
- Feel free to open issues or pull requests. Suggested workflow:
	1. Fork the repo and create a feature branch.
	2. Add tests for new functionality.
	3. Submit a PR with a clear description of changes.

Checklist for PRs:
- Code passes linting and tests.
- No sensitive data checked in.
- README updated when behavior changes.

License
Specify your license here (e.g., MIT). If you don't have a preferred license yet, consider adding an appropriate `LICENSE` file.

Next steps & improvements
- Add automated tests and a CI pipeline (GitHub Actions) to run linting and tests.
- Add scripts to export model metrics and evaluation plots to `logs/` or `reports/`.
- Containerize the app with a `Dockerfile` for reproducible deployments.

Contact
For questions, contact the repo owner or open an issue.
