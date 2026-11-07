"""LAB 09: Run batch inference from a CSV of AQI features."""
from _common import ROOT
import sys
sys.path.insert(0,str(ROOT))
from predict import predict
from labs._common import lab_start, lab_complete
import argparse
if __name__ == '__main__':
    lab_start(9, 'Batch Inference')
    p=argparse.ArgumentParser(); p.add_argument('--input',default='sample_input.csv'); p.add_argument('--output',default='artifacts/lab09_predictions.csv'); a=p.parse_args()
    predict(a.input,a.output)
    lab_complete(9, 'Batch Inference')
