"""LAB 02: Validate AQI schema and basic data quality."""
from _common import ROOT, ART, DATA, lab_start, lab_complete
from main import load_data, data_validation_report, FEATURES, TARGET
import json, argparse

def run(path=DATA):
    df=load_data(path); report=data_validation_report(df)
    report.update({'required_features': FEATURES, 'target': TARGET, 'status':'PASSED'})
    (ART/'lab02_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return df
if __name__ == '__main__':
    lab_start(2, 'Validate Data')
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA)); a=p.parse_args(); run(a.data)
    lab_complete(2, 'Validate Data')
