# Air Quality Prediction MLOps — Lab-wise Programs

This is the AQI project organized into individually executable numbered lab programs. Each lab has its own Python file under `labs/`; do not copy the churn prediction code into this AQI project. The AQI task remains a **regression** task, so evaluation uses RMSE, MAE and R².

## 1. Setup in VS Code (Windows)

1. Extract the ZIP and open the extracted `aqi_labwise_work` folder in VS Code.
2. Open **Terminal → New Terminal** and confirm the terminal is at the project root (the folder containing `main.py`).
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
5. The `data/city_day.csv` dataset is included in this package. Run each lab from the project root as shown below.

## 2. Lab list and run commands

| Lab | Program | Purpose | Command |
|---|---|---|---|
| 01 | `labs/lab01_load_data.py` | Load data, inspect rows/columns/missing values, record SHA-256 lineage | `python labs/lab01_load_data.py` |
| 02 | `labs/lab02_validate_data.py` | Validate required AQI columns and clean usable records | `python labs/lab02_validate_data.py` |
| 03 | `labs/lab03_preprocessing.py` | Train/test split and leakage-safe numeric/categorical preprocessing demonstration | `python labs/lab03_preprocessing.py` |
| 04 | `labs/lab04_validate_outputs.py` | Check that model, metadata and reports exist and are valid | `python labs/lab04_validate_outputs.py` |
| 05 | `labs/lab05_train_model.py` | Train the Random Forest regression pipeline and save model/metrics | `python labs/lab05_train_model.py` |
| 06 | `labs/lab06_model_evaluation.py` | Summarize RMSE, MAE, R² and residuals | `python labs/lab06_model_evaluation.py` |
| 07 | `labs/lab07_quality_gate.py` | Check model metrics against `minimum_r2` in `config.json` | `python labs/lab07_quality_gate.py` |
| 08 | `labs/lab08_automate_lifecycle.py` | Promote/copy the model only when the saved quality gate passes | `python labs/lab08_automate_lifecycle.py` |
| 09 | `labs/lab09_inference.py` | Run predictions from a feature CSV | `python labs/lab09_inference.py --input sample_input.csv` |
| 10 | `labs/lab10_api_serving.py` | FastAPI health, metadata, single and batch prediction endpoints | `python -m uvicorn labs.lab10_api_serving:app --reload` |
| 11 | `labs/lab11_prepare_deployment.py` | Prepare a local deployment bundle | `python labs/lab11_prepare_deployment.py` |
| 12 | `labs/lab12_monitoring.py` | API latency report and optional feature drift checks | `python labs/lab12_monitoring.py` |

**Correction for Lab 07 command:** run `python labs/lab07_quality_gate.py`.

## 3. Recommended execution order

Run in this order when demonstrating the complete lifecycle:

```powershell
python labs/lab01_load_data.py
python labs/lab02_validate_data.py
python labs/lab03_preprocessing.py
python labs/lab05_train_model.py
python labs/lab06_model_evaluation.py
python labs/lab07_quality_gate.py
python labs/lab08_automate_lifecycle.py
python labs/lab09_inference.py --input sample_input.csv
```

Lab 04 checks the artifacts produced by Lab 05, so run it after training if you want the checks to pass. Lab 10 is a long-running server; leave its terminal open, then open `http://127.0.0.1:8000/docs` in your browser to test the API. In another terminal, run Lab 12 after sending API requests. To also check feature drift, provide a CSV containing all feature columns:

```powershell
python labs/lab12_monitoring.py --input path\to\current_observations.csv --threshold 0.25
```

## 4. Lab 10 API request example

Open `http://127.0.0.1:8000/docs`, select `POST /predict`, and send JSON shaped like this. All 14 feature names are required:

```json
{
  "values": {
    "PM2.5": 80, "PM10": 120, "NO": 10, "NO2": 25,
    "NOx": 30, "NH3": 8, "CO": 1.2, "SO2": 12,
    "O3": 40, "Benzene": 1.1, "Toluene": 4.5, "Xylene": 2.0,
    "City": "Delhi", "Season": "Winter"
  }
}
```

For `POST /predict/batch`, send `{"records": [{...}, {...}]}` with each record containing the same 14 features. Other endpoints: `GET /health` and `GET /metadata`.

## 5. Important notes

- Training (`Lab 05`) is the source of truth for the current model and regenerates model artifacts. It can take a while because it trains a Random Forest on the included dataset.
- `Lab 03` saves split CSVs under `artifacts/`; the production training pipeline uses the same preprocessing strategy inside a scikit-learn pipeline.
- `Lab 08` uses a simple local `production/` folder as a lifecycle demonstration; it does not claim to create an MLflow registry.
- `Lab 10` loads the saved local model from `models/aqi_model.joblib`. Run Lab 05 first if the model is missing or you want to retrain.
- `Lab 12` writes performance summaries to `artifacts/`. Drift detection uses the project's lightweight heuristic and is a warning signal, not a statistical proof of drift.
- The ZIP excludes the virtual environment and redundant large hourly datasets to keep the download manageable. The included `data/city_day.csv` is sufficient for the numbered labs.


## Execution confirmation messages

Every numbered lab prints a banner such as `EXECUTING LAB 03: Preprocessing` before it runs and `LAB 03 EXECUTED SUCCESSFULLY: Preprocessing` after it finishes successfully. If a lab raises an error, the success message will not appear; use the error printed above it to troubleshoot. Lab 10 prints its Lab 10 banner when the API server starts and remains running until you stop it with Ctrl+C.
