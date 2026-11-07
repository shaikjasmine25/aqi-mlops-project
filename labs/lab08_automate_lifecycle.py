"""LAB 08: Automate lifecycle decision and promote only a passing model."""
from _common import ROOT, ART, MODELS, lab_start, lab_complete
import json, shutil

def run():
    gate_path=ART/'quality_gate_report.json'
    model=MODELS/'aqi_model.joblib'; metadata=MODELS/'metadata.json'
    if not gate_path.exists() or not model.exists(): raise FileNotFoundError('Run Lab 05 first.')
    gate=json.loads(gate_path.read_text()); passed=bool(gate.get('passed'))
    production=ROOT/'production'; production.mkdir(exist_ok=True)
    if passed:
        shutil.copy2(model,production/'aqi_model.joblib')
        if metadata.exists(): shutil.copy2(metadata,production/'metadata.json')
        decision='PROMOTED'
    else: decision='NOT_PROMOTED'
    report={'lab':'08 - Automated Model Lifecycle','decision':decision,'quality_gate_passed':passed,
            'production_model':str(production/'aqi_model.joblib') if passed else None,
            'note':'Promotion copies the validated local artifact; this project does not require an MLflow registry.'}
    (ART/'lab08_lifecycle_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return report
if __name__ == '__main__':
    lab_start(8, 'Automate Model Lifecycle')
    run()
    lab_complete(8, 'Automate Model Lifecycle')
