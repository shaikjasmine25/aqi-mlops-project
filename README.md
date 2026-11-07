# AQI Prediction + MLOps (VS Code)

This project adapts the useful lifecycle concepts from the churn project to AQI regression. It keeps AQI as a **regression** task, so it uses RMSE, MAE and R² rather than churn classification metrics such as F1/ROC-AUC.

## 1. Set up in VS Code
1. Extract this ZIP and open the `aqi_vscode_mlops` folder in VS Code.
2. Put the dataset file named `city_day.csv` inside `data/`. Expected source: India Air Quality dataset.
3. Open Terminal in VS Code and create a virtual environment:
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `python -m pip install -r requirements.txt`
5. Run the whole pipeline: `python main.py --data data/city_day.csv`

The dataset is not bundled because it was not included with the uploaded notebook. If your CSV is elsewhere, pass its path, e.g. `python main.py --data "C:/Users/you/Downloads/city_day.csv"`.

## 2. Other commands
- Predict from a CSV containing all 14 features: `python predict.py --input sample_input.csv`
- Check feature drift: `python monitor.py --input new_observations.csv`
- Run tests: `pytest -q`
- Optional API: after training, run `uvicorn api:app --reload`; open `http://127.0.0.1:8000/docs`.

For `predict.py`, `sample_input.csv` must include: `PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, City, Season`.

## 3. Churn concepts and AQI equivalents
Already present in your notebook and therefore not duplicated as a second implementation: preprocessing/model pipeline, data checks, model persistence, RMSE/MAE/R², plots, MLflow tracking (optional), reproducibility check, output validation, and Git workflow.

Added here for a runnable VS Code workflow:
- repeatable command-line lifecycle (load → validate → split → train → evaluate → quality gate → save)
- metadata and dataset SHA-256 lineage
- configurable quality gate
- prediction CLI and inference history
- FastAPI `/health` and `/predict` endpoint
- lightweight feature-drift monitoring
- request latency logging
- smoke tests for season mapping and unknown categories
- logging and output reports
- `.gitignore` and setup instructions

Not copied literally: churn-specific customer IDs, churn labels, classification confusion matrix/ROC, thresholding for positive class, customer-specific schemas, Kubernetes and container deployment. These require different semantics or are unnecessary for a first runnable local AQI project. The drift checker is a simple heuristic; it is not a full statistical drift or concept-drift detector. API monitoring logs request latency, but does not yet calculate rolling service-level alerts.

## 4. Outputs
After training, check:
- `models/aqi_model.joblib`, `models/metadata.json`
- `artifacts/evaluation_metrics.json`, `quality_gate_report.json`, `reproducibility_report.json`
- `artifacts/pipeline_report.json`, `dataset_lineage.json`, `predictions.csv`
- `artifacts/actual_vs_predicted.png`, `reference_features.csv`
- `logs/pipeline.log`

## Important
The pipeline uses a fixed random seed and the same held-out split for reproducibility. Its `minimum_r2` default is 0.0, a basic guardrail; choose a stricter value only after reviewing actual results. The pipeline may take a few minutes depending on the computer and dataset.
