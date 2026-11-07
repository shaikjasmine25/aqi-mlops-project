"""LAB 04: Validate training/evaluation outputs and saved model metadata."""
from _common import ROOT, ART, MODELS, lab_start, lab_complete
import json, math

def run():
    checks=[]
    for name in ['data_validation_report.json','evaluation_metrics.json','quality_gate_report.json','reproducibility_report.json']:
        path=ART/name; checks.append({'file':name,'exists':path.exists()})
    model=MODELS/'aqi_model.joblib'; meta=MODELS/'metadata.json'
    checks += [{'file':'models/aqi_model.joblib','exists':model.exists()}, {'file':'models/metadata.json','exists':meta.exists()}]
    if (ART/'evaluation_metrics.json').exists():
        metrics=json.loads((ART/'evaluation_metrics.json').read_text())
        checks.append({'file':'evaluation_metrics.json','finite_values':all(math.isfinite(float(v)) for v in metrics.values())})
    if meta.exists():
        metadata=json.loads(meta.read_text()); checks.append({'file':'metadata.json','features_present':bool(metadata.get('features'))})
    passed=all(all(v for k,v in c.items() if k!='file') for c in checks)
    report={'lab':'04 - Output Validation','status':'PASSED' if passed else 'NEEDS_ATTENTION','checks':checks}
    (ART/'lab04_output_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return passed
if __name__ == '__main__':
    lab_start(4, 'Validate Outputs')
    run()
    lab_complete(4, 'Validate Outputs')
