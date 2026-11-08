# AQI Prediction + MLOps (VS Code)

This project implements an end-to-end Air Quality Index (AQI) prediction system using Machine Learning and MLOps practices. It focuses on predicting AQI values as a **regression task**, using evaluation metrics such as RMSE, MAE, and R².

## 1. Set Up in VS Code

1. Extract the project ZIP file and open the `aqi_vscode_mlops` folder in VS Code.
2. Place the dataset file named `city_day.csv` inside the `data/` folder. The dataset is based on the India Air Quality dataset.
3. Open the terminal in VS Code and create a virtual environment:

   - **Windows:**
     ```bash
     py -m venv .venv
     .venv\Scripts\activate
     ```

   - **macOS/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the complete AQI prediction pipeline:

   ```bash
   python main.py --data data/city_day.csv
   ```

**Note:** The dataset is not bundled with the project. If your CSV file is stored elsewhere, provide its path:

```bash
python main.py --data "C:/Users/you/Downloads/city_day.csv"
```

## 2. Other Commands

- **Make predictions from a CSV file:**
  ```bash
  python predict.py --input sample_input.csv
  ```

- **Check feature drift:**
  ```bash
  python monitor.py --input new_observations.csv
  ```

- **Run tests:**
  ```bash
  pytest -q
  ```

- **Start the optional FastAPI application** (after training):
  ```bash
  uvicorn api:app --reload
  ```

  Open the API documentation in your browser:

  `http://127.0.0.1:8000/docs`

For `predict.py`, the `sample_input.csv` file must contain these 14 features:

`PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, City, Season`

## 3. MLOps Features

The project includes the following Machine Learning and MLOps components:

- Data preprocessing and validation
- Missing-value handling
- Feature engineering
- Train-test splitting
- Machine Learning model training
- Regression model evaluation using RMSE, MAE, and R²
- Data and output quality checks
- Model persistence and versioning
- MLflow experiment tracking (optional)
- Reproducibility checks
- Dataset SHA-256 hash and lineage tracking
- Configurable model quality gate
- Command-line prediction interface
- Prediction history tracking
- FastAPI `/health` and `/predict` endpoints
- Feature-drift monitoring
- API request latency logging
- Automated smoke tests
- Pipeline logging and output reports
- Git and GitHub version control
- GitHub Actions continuous integration (CI)

## 4. Project Outputs

After successfully running the training pipeline, the following files and folders are generated:

**Models:**
- `models/aqi_model.joblib`
- `models/metadata.json`

**Evaluation and Reports:**
- `artifacts/evaluation_metrics.json`
- `artifacts/quality_gate_report.json`
- `artifacts/reproducibility_report.json`
- `artifacts/pipeline_report.json`
- `artifacts/dataset_lineage.json`
- `artifacts/predictions.csv`

**Visualizations and Reference Data:**
- `artifacts/actual_vs_predicted.png`
- `artifacts/reference_features.csv`

**Logs:**
- `logs/pipeline.log`

## 5. Model Evaluation

Since AQI prediction is a regression problem, the project uses the following evaluation metrics:

| Metric | Purpose |
|---|---|
| RMSE | Measures the square root of the average squared prediction error. |
| MAE | Measures the average absolute prediction error. |
| R² Score | Measures how well the model explains the variation in AQI values. |

The pipeline uses a fixed random seed and a consistent held-out test split to support reproducibility.

The default `minimum_r2` quality-gate threshold is `0.0`. This is a basic guardrail. A stricter threshold should be selected after reviewing the actual model evaluation results.

## 6. Important Notes

- The project is designed for AQI regression and predicts continuous AQI values.
- The dataset must be placed inside the `data/` directory before running the pipeline.
- The pipeline may take a few minutes to execute depending on the computer and dataset size.
- MLflow tracking is optional.
- The feature-drift checker uses a lightweight heuristic and is not a complete statistical or concept-drift detection system.
- API monitoring records request latency but does not currently generate rolling service-level alerts.

## 7. GitHub Repository

The project uses Git for version control and GitHub Actions for automated testing.

Repository: https://github.com/shaikjasmine25/aqi-mlops-project