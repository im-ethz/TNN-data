import pandas as pd
import numpy as np

from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

from config import rider_mapping, DATA_PATH

root = DATA_PATH+'calendar/'

riders = ('oliver-behringer',
          'andrea-peron',
          'stephen-clancy',
          'declan-irvine',
          'brian-kamstra',
          'samuel-munday',
          'umberto-poli',
          'hamish-beadle',
          'mehdi-benhamouda',
          'sam-brand',
          'gerd-de-keijzer',
          'david-lozano',
          'peter-kusztor',
          'joonas-henttala',
          'charles-planet',
          'ulugbek-saidov',
          'lucas-dauge',
          'logan-phippen')

years = (2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021)

user_agent = {'User-agent':'Chrome/86.0.4240.22'}
base_url = 'https://www.procyclingstats.com/rider/'

df = {}
for i in tqdm(riders):
    df[i] = {}
    for y in years:
        url = base_url + f'{i}/{y}'
        response = requests.get(url, headers=user_agent)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table')
            if table:
                headers = [h.text for h in table.find_all('th')]
                data = [[r.text for r in row.find_all('td')] for row in table.find_all('tr')[1:]]
                df[i][y] = pd.DataFrame(data, columns=headers)

df = pd.concat({i: pd.concat(table) for i, table in df.items()})
df = df.replace({'':np.nan})

# reset index
df = df.reset_index()
df = df.rename(columns={'level_0':'RIDER', 'level_1':'year'})
df = df.drop('level_2', axis=1)

# fix rider
df['RIDER'] = df['RIDER'].apply(lambda x: rider_mapping[' '.join(x.split('-')[1:])])

# get date
df = df.dropna(subset=['Date']) # drop rows like youth classification
df['date'] = df.apply(lambda x: pd.to_datetime(f"{x['Date']}.{x['year']}", dayfirst=True), axis=1)
df = df.drop(['Date', 'year'], axis=1)

# drop useless columns
df = df.drop(['', 'Distance', 'PointsPCS', 'PointsUCI'], axis=1)

# get name of tour
df['raceid'] = (df['Result'].isna() | (~df['Race'].str[:5].isin(['Stage', 'Prolo']))).cumsum()
df_tours = df.groupby('raceid')['Race'].first().reset_index().rename(columns={'Race':'tour'})
df = pd.merge(df, df_tours, on=['raceid'], how='left')
df = df[df['Result'].notna()] # drop race row with tour title
df = df.drop('raceid', axis=1)

df = df.rename(columns={'Race':'race', 'Result':'result'})
df = df.sort_values(['RIDER', 'date'])
df = df.reset_index(drop=True)

# save tour file
df.drop_duplicates(subset=['tour'], keep='first')[['RIDER', 'date', 'tour']].to_csv(root+'/tour_countries.csv')

##### manually assign countries to tours
df_countries = pd.read_csv(root+'/tour_countries.csv', index_col=0)
df_countries = df_countries.set_index('tour')['country'].to_dict()

df['country'] = df['tour'].map(df_countries)

df = df[['RIDER', 'date', 'country', 'tour', 'race', 'result']]
df.to_csv(root+'/procyclingstats.csv')