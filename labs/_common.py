"""Shared helpers for numbered AQI MLOps labs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / 'artifacts'
MODELS = ROOT / 'models'
DATA = ROOT / 'data' / 'city_day.csv'
ART.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def lab_start(number, title):
    print('\n' + '=' * 60)
    print(f'EXECUTING LAB {number:02d}: {title}')
    print('=' * 60, flush=True)

def lab_complete(number, title):
    print('\n' + '-' * 60)
    print(f'LAB {number:02d} EXECUTED SUCCESSFULLY: {title}')
    print('-' * 60, flush=True)
