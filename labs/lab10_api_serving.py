"""LAB 10: FastAPI model serving with health, metadata, single and batch prediction."""
try:
    from labs._common import ROOT, ART, MODELS, lab_start, lab_complete
except ModuleNotFoundError:
    from _common import ROOT, ART, MODELS, lab_start, lab_complete
import json, time, math
from datetime import datetime, timezone
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from main import FEATURES

app=FastAPI(title='AQI Prediction API - Lab 10', version='1.0.0', description='AQI regression model serving')
_MODEL=None

@app.on_event('startup')
async def lab10_startup_message():
    lab_start(10, 'FastAPI Model Serving')
    print('LAB 10 API SERVER IS RUNNING. Open http://127.0.0.1:8000/docs to test it.', flush=True)

def get_model():
    global _MODEL
    path=MODELS/'aqi_model.joblib'
    if not path.exists(): raise HTTPException(status_code=503,detail='Model unavailable. Run Lab 05 first.')
    if _MODEL is None: _MODEL=joblib.load(path)
    return _MODEL

class PredictionRequest(BaseModel):
    values: dict[str, Any] = Field(..., description='Feature names and values required by the AQI model')
class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)

def validate_records(records):
    cleaned=[]
    for index, record in enumerate(records):
        missing=[c for c in FEATURES if c not in record]
        if missing: raise HTTPException(status_code=422, detail={'record_index':index,'missing_features':missing})
        row={c:record[c] for c in FEATURES}
        for c in FEATURES[:-2]:
            try: row[c]=float(row[c])
            except (TypeError,ValueError): raise HTTPException(status_code=422,detail=f'Feature {c} must be numeric (record {index}).')
            if not math.isfinite(row[c]): raise HTTPException(status_code=422,detail=f'Feature {c} must be finite (record {index}).')
        for c in ['City','Season']:
            if not isinstance(row[c],str) or not row[c].strip(): raise HTTPException(status_code=422,detail=f'{c} must be a non-empty string (record {index}).')
        cleaned.append(row)
    return pd.DataFrame(cleaned,columns=FEATURES)

def log_record(record):
    with (ART/'api_requests.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')

@app.get('/health')
def health():
    return {'status':'ok','model_available':(MODELS/'aqi_model.joblib').exists()}

@app.get('/metadata')
def metadata():
    path=MODELS/'metadata.json'
    if not path.exists(): raise HTTPException(status_code=404,detail='Model metadata not found; run Lab 05 first.')
    return json.loads(path.read_text(encoding='utf-8'))

@app.post('/predict')
def predict_one(request: PredictionRequest):
    started=time.perf_counter(); frame=validate_records([request.values]); model=get_model()
    try: value=float(model.predict(frame)[0])
    except Exception as exc: raise HTTPException(status_code=400,detail=f'Prediction failed: {type(exc).__name__}')
    result={'timestamp':datetime.now(timezone.utc).isoformat(),'predicted_aqi':value,'latency_ms':round((time.perf_counter()-started)*1000,2)}
    log_record({'status':'success',**result}); return result

@app.post('/predict/batch')
def predict_batch(request: BatchPredictionRequest):
    started=time.perf_counter(); frame=validate_records(request.records); model=get_model()
    try: values=[float(x) for x in model.predict(frame)]
    except Exception as exc: raise HTTPException(status_code=400,detail=f'Batch prediction failed: {type(exc).__name__}')
    result={'timestamp':datetime.now(timezone.utc).isoformat(),'count':len(values),'predicted_aqi':values,'latency_ms':round((time.perf_counter()-started)*1000,2)}
    log_record({'status':'success','timestamp':result['timestamp'],'count':len(values),'latency_ms':result['latency_ms']}); return result

if __name__=='__main__':
    import uvicorn
    uvicorn.run('labs.lab10_api_serving:app',host='127.0.0.1',port=8000,reload=True)
