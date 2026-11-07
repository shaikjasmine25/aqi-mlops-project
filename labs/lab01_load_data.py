"""LAB 01: Load AQI data, inspect it, and record dataset lineage."""
from _common import ROOT, ART, DATA, lab_start, lab_complete
import hashlib, json, argparse
import pandas as pd
from pathlib import Path

def run(path=DATA):
    path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not path.exists(): raise FileNotFoundError(f'Dataset not found: {path}')
    df = pd.read_csv(path)
    report = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'rows': int(len(df)), 'columns': list(df.columns), 'duplicate_rows': int(df.duplicated().sum()),
              'missing_cells': int(df.isna().sum().sum())}
    (ART/'lab01_data_inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2)); print('\nFirst 5 rows:'); print(df.head().to_string(index=False))
    return df
if __name__ == '__main__':
    lab_start(1, 'Load Data')
    p=argparse.ArgumentParser(); p.add_argument('--data', default=str(DATA)); a=p.parse_args(); run(a.data)
    lab_complete(1, 'Load Data')
