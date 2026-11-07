"""LAB 07: Apply a configurable R-squared quality gate."""
from _common import ROOT, ART, lab_start, lab_complete
import json, math, sys

def run():
    metric_path=ART/'evaluation_metrics.json'
    if not metric_path.exists(): raise FileNotFoundError('Run Lab 05 first: python labs/lab05_train_model.py')
    config_path=ROOT/'config.json'; config=json.loads(config_path.read_text()) if config_path.exists() else {'minimum_r2':0.0}
    metrics=json.loads(metric_path.read_text()); threshold=float(config.get('minimum_r2',0.0))
    finite=all(math.isfinite(float(v)) for v in metrics.values()); passed=finite and float(metrics['r2'])>=threshold
    report={'lab':'07 - Model Quality Gate','metric':'r2','minimum_value':threshold,'actual_value':metrics.get('r2'),
            'finite_metrics':finite,'status':'PASSED' if passed else 'FAILED'}
    (ART/'lab07_quality_gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return passed
if __name__ == '__main__':
    lab_start(7, 'Model Quality Gate')
    sys.exit(0 if run() else 1)
    lab_complete(7, 'Model Quality Gate')
