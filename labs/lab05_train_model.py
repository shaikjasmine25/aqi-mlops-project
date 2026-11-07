"""LAB 05: Train and persist the AQI regression pipeline."""
from _common import ROOT, DATA, lab_start, lab_complete
from main import run
import argparse
if __name__ == '__main__':
    lab_start(5, 'Train Model')
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA)); a=p.parse_args()
    summary=run(a.data)
    print('\nLAB 05 complete: model and training artifacts saved.')
    lab_complete(5, 'Train Model')
