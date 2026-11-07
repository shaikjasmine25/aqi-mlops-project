"""LAB 06: Summarize AQI regression metrics and residual diagnostics."""
from _common import ART, lab_start, lab_complete
import json, pandas as pd, numpy as np

def run():
    metrics_path=ART/'evaluation_metrics.json'; pred_path=ART/'predictions.csv'
    if not metrics_path.exists() or not pred_path.exists(): raise FileNotFoundError('Run Lab 05 first: python labs/lab05_train_model.py')
    metrics=json.loads(metrics_path.read_text()); df=pd.read_csv(pred_path)
    residual=df['residual'].dropna()
    report={'lab':'06 - Model Evaluation','metrics':metrics,'prediction_rows':len(df),
            'mean_residual':float(residual.mean()),'residual_std':float(residual.std()),
            'max_absolute_error':float(residual.abs().max()),
            'interpretation':'RMSE and MAE are error measures (lower is better); R2 is goodness-of-fit (closer to 1 is better).'}
    (ART/'lab06_evaluation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return report
if __name__ == '__main__':
    lab_start(6, 'Model Evaluation')
    run()
    lab_complete(6, 'Model Evaluation')
