import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

from plot import *
from helper import *
from config import rider_mapping

path = './'
rider_mapping = {k.upper() : v for k, v in rider_mapping.items()}

df = pd.read_excel(path+'TEST ANALYSIS Dec_2018.xlsx', nrows=16, header=(0,1), sheet_name=None)

# make sure all tabs are using the same units
df['Dec_2018'].loc[:,('VT1 (GET)', 'VO2%max')] /= 100
df['Dec_2018'].loc[:,('VT1 (GET)', '%HRmax')] /= 100
df['Dec_2018'].loc[:,('VT1 (GET)', '%Wmax')] /= 100
df['Dec_2018'].loc[:,('VT2 (RCP)', 'VO2%max')] /= 100
df['Dec_2018'].loc[:,('VT2 (RCP)', '%HRmax')] /= 100
df['Dec_2018'].loc[:,('VT2 (RCP)', '%Wmax')] /= 100

# combine all tabs
df = pd.concat(df)

# clean up columns
df.drop([('Min', 'La.1'), ('Min', 'La.2'), ('Min', 'La.3'), ('Min', 'La.4'), ('Min', 'La.5'), ('Min', 'La.6')], axis=1, inplace=True)
df = df[['ID and ANTROPOMETRY', 'SPIROMETRY', 'VT1 (GET)', 'VT2 (RCP)', 'VO2peak',
		'EFFICIENCY', 'LT1', 'LT2', 'MAP', 'MLSS', "60' power", 'Baseline', 'Min']]
df.dropna(how='all', axis=1, inplace=True)

# apply anonymous mapping riders
rider_mapping.update({'BEHRINGHER':1})
df['RIDER'] = df[('ID and ANTROPOMETRY', 'Surname')].map(rider_mapping)
df.drop([('ID and ANTROPOMETRY', 'Name'), ('ID and ANTROPOMETRY', 'Surname')], axis=1, inplace=True)

# reset index
df = df.reset_index()
df.drop('level_1', axis=1, inplace=True)
df.rename(columns={'level_0':'date'}, inplace=True)
df = df.set_index(['RIDER', 'date']).sort_index()

df.to_csv(path+'fitness_analysis_2019_anonymous.csv')