import numpy as np 
import pandas as pd
import datetime

import sys
sys.path.append('../../')

from config import rider_mapping

df = pd.read_excel('./birthday_diagnosis.xlsx')
df.drop('Unnamed: 4', axis=1, inplace=True)
df.rename(columns={'Rider':'RIDER', 'Birthday':'dob', 'Age':'age', 'Age Diagnosis':'age_diagnosis'}, inplace=True)

# rider
df['RIDER'] = df['RIDER'].str.split().apply(lambda x: ' '.join(x[1:])).str.lower()
df['RIDER'] = df['RIDER'].map(rider_mapping)
df.sort_values('RIDER', inplace=True)
df.reset_index(drop=True, inplace=True)

# birthday
df['dob'] = pd.to_datetime(df['dob'])

# age on 2019-01-01
df['age'] = ((datetime.datetime(2019,1,1) - df['dob']).dt.days / 365.25).astype(int)

# diabetes duration
df['diabetes_duration'] = df['age'] - df['age_diagnosis']

df.drop(['dob', 'age_diagnosis'], axis=1, inplace=True)

df.to_csv('./age_diagnosis.csv', index_label=False)