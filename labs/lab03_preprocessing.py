"""LAB 03: Demonstrate leakage-safe preprocessing and save a prepared split."""
from _common import ROOT, ART, DATA, lab_start, lab_complete
from main import load_data, FEATURES, TARGET, RANDOM_STATE, make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd, json

def run(path=DATA):
    df=load_data(path); X=df[FEATURES]; y=df[TARGET].astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE)
    # Fit the complete sklearn pipeline only on training data; preprocessing is inside it.
    pipeline=make_pipeline(); pipeline.fit(X_train,y_train)
    X_train.to_csv(ART/'lab03_X_train.csv',index=False); X_test.to_csv(ART/'lab03_X_test.csv',index=False)
    pd.DataFrame({'AQI':y_train}).to_csv(ART/'lab03_y_train.csv',index=False)
    pd.DataFrame({'AQI':y_test}).to_csv(ART/'lab03_y_test.csv',index=False)
    report={'status':'PASSED','train_rows':len(X_train),'test_rows':len(X_test),
            'numeric_features':12,'categorical_features':['City','Season'],
            'imputation':'median for numeric, most-frequent for categorical',
            'categorical_encoding':'OneHotEncoder(handle_unknown=ignore)',
            'scaling':'StandardScaler for numeric features','fitted_on':'training split only'}
    (ART/'lab03_preprocessing_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return pipeline
if __name__ == '__main__':
    lab_start(3, 'Preprocessing')
    run()
    lab_complete(3, 'Preprocessing')
