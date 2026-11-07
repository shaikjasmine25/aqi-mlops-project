"""LAB 12: Analyze inference API latency and run feature-drift monitoring."""
from _common import ROOT, ART, lab_start, lab_complete
import json, argparse, pandas as pd
from monitor import monitor

def performance_report():
    path=ART/'api_requests.jsonl'
    if not path.exists():
        report={'status':'NO_API_LOGS','message':'Start Lab 10 and send predictions before measuring API performance.'}
    else:
        rows=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
        df=pd.DataFrame(rows); lat=pd.to_numeric(df.get('latency_ms',pd.Series(dtype=float)),errors='coerce').dropna()
        report={'total_logged_requests':len(df),'successful_requests':int((df.get('status',pd.Series(dtype=str))=='success').sum()),
                'latency_ms':{'mean':float(lat.mean()) if len(lat) else None,'median':float(lat.median()) if len(lat) else None,
                              'p95':float(lat.quantile(.95)) if len(lat) else None,'max':float(lat.max()) if len(lat) else None}}
    (ART/'lab12_api_performance_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('API performance:'); print(json.dumps(report,indent=2)); return report

if __name__ == '__main__':
    lab_start(12, 'API Performance and Drift Monitoring')
    p=argparse.ArgumentParser(); p.add_argument('--input',help='Optional CSV with current AQI feature observations'); p.add_argument('--threshold',type=float,default=.25); a=p.parse_args()
    performance_report()
    if a.input:
        print('\nFeature drift:'); monitor(a.input,threshold=a.threshold)
    else: print('\nFor feature drift, add --input path/to/current_observations.csv')
    lab_complete(12, 'API Performance and Drift Monitoring')
