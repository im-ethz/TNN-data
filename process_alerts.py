import pandas as pd
import numpy as np

import glob
from tqdm import tqdm

from config import rider_mapping, DATA_PATH

df = {}
for source in ('EU', 'US'):
    for file in tqdm(glob.glob(f"{DATA_PATH}/Dexcom/export/{source}/*")):
        df_i = pd.read_csv(file, nrows=20, index_col=0)
        if len(df_i) <= 3:
            continue
        firstname = df_i.loc[df_i['Event Type'] == 'FirstName', 'Patient Info'].item().lower()
        lastname = df_i.loc[df_i['Event Type'] == 'LastName', 'Patient Info'].item().lower()
        try:
            ts = pd.to_datetime(df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'].dropna().iloc[0])
        except IndexError:
            continue

        df_i[df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'].isna()]
        df_i = df_i[df_i['Event Type'] == 'Alert']
        df_i = df_i.drop(['Timestamp (YYYY-MM-DDThh:mm:ss)', 'Device Info', 'Insulin Value (u)',
                          'Carb Value (grams)', 'Transmitter Time (Long Integer)', 'Transmitter ID', 
                          'Patient Info', 'Event Type'], axis=1)
        df_i = df_i.reset_index(drop=True)
        df_i = df_i.set_index(['Source Device ID', 'Event Subtype'])
        df[(firstname, lastname, ts, source)] = df_i
df = pd.concat(df, names=['firstname', 'lastname', 'timestamp', 'source'])

df = df.reset_index()
df['lastname'] = df['lastname'].replace({'declan':'irvine', 'clancey':'clancy'})

df['RIDER'] = df['lastname'].map(rider_mapping)
df = df.drop(['firstname', 'lastname'], axis=1)
df = df.set_index(['RIDER', 'timestamp', 'Source Device ID', 'source', 'Event Subtype'])
df = df.sort_index()

df[df.columns.drop('Duration (hh:mm:ss)')] = df[df.columns.drop('Duration (hh:mm:ss)')].astype(float)
df['Glucose Value (mmol/L)'] = df['Glucose Value (mmol/L)'].fillna((df['Glucose Value (mg/dL)'] / 18).round(1))
df['Glucose Rate of Change (mmol/L/min)'] = df['Glucose Rate of Change (mmol/L/min)'].fillna((df['Glucose Rate of Change (mg/dL/min)'] / 18).round(1))
df = df.drop(['Glucose Value (mg/dL)', 'Glucose Rate of Change (mg/dL/min)'], axis=1)

df.to_csv(f"{DATA_PATH}/Dexcom/dexcom_alerts.csv")

# select only relevant data
RIDERS = [1, 2, 3, 4, 5, 6, 10, 12, 13, 14, 15, 16]
# Note: this is a result from the other repo. We just hardcode it here to save time.
# If something is changed in the other repo, this list should also be updated here.

dates = pd.read_csv(f'{DATA_PATH}/calendar/season_dates.csv', index_col=[0,1], header=[0,1])
dates.index.names = ['RIDER', 'name']
dates.index = dates.index.droplevel('name')
dates = dates.loc[RIDERS]
dates = dates['2019']
dates['start'] = pd.to_datetime(dates['start'])
dates['end'] = pd.to_datetime(dates['end'])

df = df.loc[RIDERS]
df = df.reset_index()
df['before'] = df.apply(lambda x: x['timestamp'] < dates.loc[x['RIDER'], 'start'], axis=1)
df['after'] = df.apply(lambda x: x['timestamp'] > dates.loc[x['RIDER'], 'end'], axis=1)

df = df[~df['before'] & ~df['after']]
df = df.drop_duplicates(subset=df.columns.drop(['timestamp', 'source']))
df = df.drop(['source', 'before', 'after'], axis=1)

df = df.drop_duplicates(subset=df.columns.drop(['timestamp', 'Source Device ID']))
df = df[~df['Event Subtype'].isin(['Signal Loss', 'Urgent Low', 'Urgent Low Soon'])]
df = df.drop('Duration (hh:mm:ss)', axis=1)
df = df.set_index(['RIDER', 'timestamp', 'Source Device ID', 'Event Subtype'])

df = df.reset_index()
df = df[~df['Event Subtype'].isin(['Rise', 'Fall'])]
df = df.drop('Glucose Rate of Change (mmol/L/min)', axis=1)
df = df.set_index(['RIDER', 'timestamp', 'Source Device ID', 'Event Subtype'])
df.index = df.index.swaplevel('Event Subtype', 'timestamp')
df = df.sort_index()