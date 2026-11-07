"""Run the AQI end-to-end MLOps pipeline: python main.py --data data/city_day.csv"""
from pathlib import Path
import argparse, json, logging, hashlib, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
for folder in (ART, MODELS, LOGS):
    folder.mkdir(exist_ok=True)

POLLUTANTS = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
FEATURES = POLLUTANTS + ['City', 'Season']
TARGET = 'AQI'
RANDOM_STATE = 42
logging.basicConfig(filename=LOGS/'pipeline.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def season_from_month(month):
    if month in (12, 1, 2): return 'Winter'
    if month in (3, 4, 5): return 'Summer'
    if month in (6, 7, 8, 9): return 'Monsoon'
    return 'Post-Monsoon'

def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put city_day.csv in the data folder.")
    df = pd.read_csv(path)
    required = set(POLLUTANTS + ['City', TARGET])
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError(f"Dataset missing required columns: {missing}")
    if 'Season' not in df.columns:
        if 'Date' not in df.columns: raise ValueError("Dataset needs either Season or Date column.")
        dates = pd.to_datetime(df['Date'], errors='coerce')
        df['Season'] = dates.dt.month.map(lambda m: season_from_month(int(m)) if pd.notna(m) else np.nan)
    # A target is required for supervised training.
    df = df.dropna(subset=[TARGET, 'City', 'Season']).drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df = df.dropna(subset=[TARGET])
    if df.empty: raise ValueError("No rows with a valid AQI target remain.")
    if (df[TARGET] < 0).any(): raise ValueError("AQI target contains negative values.")
    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if np.isinf(df[col]).any(): df.loc[np.isinf(df[col]), col] = np.nan
    df['City'] = df['City'].astype(str)
    df['Season'] = df['Season'].astype(str)
    return df

def data_validation_report(df):
    report = {'status':'PASSED', 'rows':int(len(df)), 'columns':int(len(df.columns)),
              'target':'AQI', 'missing_feature_values':int(df[FEATURES].isna().sum().sum()),
              'negative_aqi_rows':int((df[TARGET] < 0).sum())}
    (ART/'data_validation_report.json').write_text(json.dumps(report, indent=2))
    return report

def make_pipeline():
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('num', numeric, POLLUTANTS), ('cat', categorical, ['City','Season'])])
    model = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([('preprocessor', prep), ('model', model)])

def metrics_for(y, pred):
    return {'rmse':float(np.sqrt(mean_squared_error(y,pred))),
            'mae':float(mean_absolute_error(y,pred)), 'r2':float(r2_score(y,pred))}

def run(data_path):
    started=time.time()
    df=load_data(data_path)
    data_validation_report(df)
    X=df[FEATURES].copy(); y=df[TARGET].astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE)
    model=make_pipeline()
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    metrics=metrics_for(y_test,pred)
    # A configurable quality gate. R2 threshold is deliberately conservative and can be changed in config.json.
    config_path=ROOT/'config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {'minimum_r2':0.0}
    gate={'passed':bool(np.isfinite(list(metrics.values())).all() and metrics['r2'] >= config.get('minimum_r2',0.0)),
          'minimum_r2':config.get('minimum_r2',0.0), 'metrics':metrics}
    (ART/'quality_gate_report.json').write_text(json.dumps(gate,indent=2))
    if not np.isfinite(list(metrics.values())).all(): raise ValueError("Quality gate failed: invalid metrics.")
    joblib.dump(model, MODELS/'aqi_model.joblib')
    metadata={'features':FEATURES,'target':TARGET,'model':'RandomForestRegressor pipeline',
              'random_state':RANDOM_STATE,'train_rows':len(X_train),'test_rows':len(X_test),
              'data_file':str(data_path),'data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    (MODELS/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (ART/'evaluation_metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({'actual_aqi':y_test.to_numpy(),'predicted_aqi':pred,
                  'residual':y_test.to_numpy()-pred}).to_csv(ART/'predictions.csv',index=False)
    plt.figure(figsize=(6,6)); plt.scatter(y_test,pred,alpha=.35)
    lo=min(float(y_test.min()),float(np.min(pred))); hi=max(float(y_test.max()),float(np.max(pred)))
    plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('Actual AQI'); plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI'); plt.tight_layout(); plt.savefig(ART/'actual_vs_predicted.png'); plt.close()
    # Reproducibility: same seed/config on same split should yield same metrics.
    model2=make_pipeline(); model2.fit(X_train,y_train); pred2=model2.predict(X_test)
    reproducible=bool(np.allclose(pred,pred2) and metrics==metrics_for(y_test,pred2))
    (ART/'reproducibility_report.json').write_text(json.dumps(
        {'status':'PASSED' if reproducible else 'FAILED','identical_predictions':reproducible,
         'run_1':metrics,'run_2':metrics_for(y_test,pred2)},indent=2))
    # Save training reference features for drift checks.
    X_train.to_csv(ART/'reference_features.csv',index=False)
    # Data lineage and lifecycle summary.
    (ART/'dataset_lineage.json').write_text(json.dumps({'source':str(data_path),'sha256':metadata['data_sha256'],
       'rows_after_validation':len(df),'feature_columns':FEATURES,'target':TARGET},indent=2))
    summary={'status':'PASSED' if reproducible and gate['passed'] else 'CHECK',
             'rows':len(df),'metrics':metrics,'quality_gate_passed':gate['passed'],
             'reproducible':reproducible,'elapsed_seconds':round(time.time()-started,2)}
    (ART/'pipeline_report.json').write_text(json.dumps(summary,indent=2))
    logging.info("Pipeline complete: %s",summary)
    print(json.dumps(summary,indent=2))
    print(f"\nModel saved to: {MODELS/'aqi_model.joblib'}")
    print(f"Reports and plots saved to: {ART}")
    if not gate['passed']:
        print("WARNING: quality gate did not pass. Adjust config.json only after reviewing metrics.")
    return summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/city_day.csv',help='Path to city_day.csv')
    args=parser.parse_args()
    run(args.data)
