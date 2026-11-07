"""Batch prediction CLI: python predict.py --input sample_input.csv"""
from pathlib import Path
import argparse, json
import pandas as pd
import joblib
from main import FEATURES, MODELS, ART

def predict(input_path, output_path=None):
    model_path=MODELS/'aqi_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError("Train first: python main.py --data data/city_day.csv")
    df=pd.read_csv(input_path)
    missing=[c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError(f"Missing input columns: {missing}")
    model=joblib.load(model_path)
    result=df[FEATURES].copy()
    result['Predicted_AQI']=model.predict(result).round(2)
    out=Path(output_path) if output_path else ART/'inference_predictions.csv'
    out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False)
    # Append inference request history for monitoring.
    history=ART/'inference_history.csv'
    result.assign(timestamp=pd.Timestamp.now(tz="UTC").isoformat()).to_csv(history,mode='a',header=not history.exists(),index=False)
    print(f"Predictions saved to {out}")
    print(result.to_string(index=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True); p.add_argument('--output')
    a=p.parse_args(); predict(a.input,a.output)
