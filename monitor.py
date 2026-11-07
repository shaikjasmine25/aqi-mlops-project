"""Simple feature/prediction drift checks against training reference data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from main import FEATURES, POLLUTANTS, ART

def monitor(current_csv, threshold=0.25):
    ref_path=ART/'reference_features.csv'
    if not ref_path.exists(): raise FileNotFoundError("Run main.py first to create reference features.")
    ref=pd.read_csv(ref_path); cur=pd.read_csv(current_csv)
    missing=[c for c in FEATURES if c not in cur.columns]
    if missing: raise ValueError(f"Monitoring data missing columns: {missing}")
    rows=[]
    # Numeric drift uses absolute relative change in means; categorical drift uses share of unseen categories.
    for col in POLLUTANTS:
        a=pd.to_numeric(ref[col],errors='coerce').dropna()
        b=pd.to_numeric(cur[col],errors='coerce').dropna()
        if len(a)==0 or len(b)==0: score=None
        else: score=float(abs(b.mean()-a.mean())/(abs(a.mean())+1e-9))
        rows.append({'feature':col,'drift_score':score,'drift_detected':bool(score is not None and score>threshold)})
    for col in ['City','Season']:
        known=set(ref[col].astype(str)); vals=cur[col].astype(str)
        score=float((~vals.isin(known)).mean())
        rows.append({'feature':col,'drift_score':score,'drift_detected':score>0})
    report={'threshold':threshold,'features_checked':len(rows),
            'drift_detected':any(r['drift_detected'] for r in rows),'details':rows,
            'note':'This is a lightweight monitoring heuristic, not a statistical proof of drift.'}
    (ART/'drift_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2)); return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--threshold',type=float,default=.25)
    a=p.parse_args(); monitor(a.input,a.threshold)
