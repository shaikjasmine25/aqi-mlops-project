"""Optional FastAPI inference endpoint. Start with: uvicorn api:app --reload"""
from pathlib import Path
import time, json
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import FEATURES, MODELS, ART

app=FastAPI(title='AQI Prediction API',version='1.0')
class AQIRequest(BaseModel):
    values: dict

@app.get('/health')
def health():
    return {'status':'ok','model_available':(MODELS/'aqi_model.joblib').exists()}

@app.post('/predict')
def predict(request:AQIRequest):
    start=time.time()
    missing=[c for c in FEATURES if c not in request.values]
    if missing: raise HTTPException(status_code=422,detail={'missing_features':missing})
    try:
        model=joblib.load(MODELS/'aqi_model.joblib')
        frame=pd.DataFrame([request.values])[FEATURES]
        value=float(model.predict(frame)[0])
        record={'timestamp':pd.Timestamp.utcnow().isoformat(),'predicted_aqi':value,
                'latency_ms':round((time.time()-start)*1000,2)}
        log=ART/'api_requests.jsonl'
        with log.open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        return record
    except FileNotFoundError: raise HTTPException(status_code=503,detail='Model not trained yet; run main.py.')
    except Exception as e: raise HTTPException(status_code=400,detail=str(e))
