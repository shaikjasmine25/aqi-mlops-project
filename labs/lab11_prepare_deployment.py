"""LAB 11: Prepare a self-contained local deployment bundle."""
from _common import ROOT, MODELS, ART, lab_start, lab_complete
from pathlib import Path
import shutil, json

def run():
    model=MODELS/'aqi_model.joblib'
    if not model.exists(): raise FileNotFoundError('Run Lab 05 to train the model before preparing deployment.')
    out=ROOT/'deployment'; out.mkdir(exist_ok=True)
    model_dir=out/'models'; model_dir.mkdir(exist_ok=True)
    labs_dir=out/'labs'; labs_dir.mkdir(exist_ok=True)
    for name in ['aqi_model.joblib','metadata.json']:
        src=MODELS/name
        if src.exists(): shutil.copy2(src,model_dir/name)
    shutil.copy2(ROOT/'requirements.txt',out/'requirements.txt')
    shutil.copy2(ROOT/'main.py',out/'main.py')
    shutil.copy2(ROOT/'labs'/'lab10_api_serving.py',labs_dir/'lab10_api_serving.py')
    shutil.copy2(ROOT/'labs'/'_common.py',labs_dir/'_common.py')
    (labs_dir/'__init__.py').write_text('',encoding='utf-8')
    report={'lab':'11 - Prepare Deployment','status':'READY','deployment_directory':str(out),
            'files':[p.name for p in out.iterdir() if p.is_file()],
            'start_command':'uvicorn labs.lab10_api_serving:app --host 127.0.0.1 --port 8000',
            'note':'Copy the deployment folder to a separate location, install its requirements, then run the start command from that folder.'}
    (ART/'lab11_deployment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return report
if __name__ == '__main__':
    lab_start(11, 'Prepare Deployment')
    run()
    lab_complete(11, 'Prepare Deployment')
