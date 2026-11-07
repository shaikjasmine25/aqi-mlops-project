import pandas as pd
from main import season_from_month, make_pipeline, FEATURES

def test_season_mapping():
    assert season_from_month(1)=='Winter'
    assert season_from_month(4)=='Summer'
    assert season_from_month(7)=='Monsoon'
    assert season_from_month(11)=='Post-Monsoon'

def test_pipeline_accepts_unknown_city():
    model=make_pipeline()
    rows=[]
    for i in range(8):
        rows.append({'PM2.5':20+i,'PM10':30+i,'NO':2,'NO2':4,'NOx':5,'NH3':3,'CO':.2,
                     'SO2':2,'O3':10,'Benzene':.1,'Toluene':.2,'Xylene':.1,
                     'City':'A' if i<4 else 'B','Season':'Winter','AQI':30+i})
    df=pd.DataFrame(rows)
    model.fit(df[FEATURES],df['AQI'])
    sample=df[FEATURES].iloc[[0]].copy(); sample['City']='NewCity'
    assert len(model.predict(sample))==1
