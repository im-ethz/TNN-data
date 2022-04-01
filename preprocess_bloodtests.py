import os

import numpy as np
import pandas as pd

from config import rider_mapping, DATA_PATH

root = DATA_PATH+'bloodtests/'

df = pd.read_excel(root+'2018-2019  TNN Riders - HbA1c.xlsx', skiprows=1, header=(0,1,2))#, nrows=16, sheet_name=None)

df = df.rename(columns={'Unnamed: 0_level_2': '', 'Unnamed: 1_level_2': ''})
df = df.dropna(subset=[('TNN HbA1c', 'first name',      '')])
df = df.drop([('TNN HbA1c', 'first name',      '')], axis=1)

# clean up name columns
df[('TNN HbA1c', 'last name', '')] = df[('TNN HbA1c', 'last name', '')].str.lower()
df[('TNN HbA1c', 'last name', '')] = df[('TNN HbA1c', 'last name', '')].apply(lambda x: x.split(' ')[0])

# anonymize
df[('TNN HbA1c', 'last name', '')] = df[('TNN HbA1c', 'last name', '')].map(rider_mapping)
df = df.dropna(subset=[('TNN HbA1c', 'last name', '')])

# clean index
df = df.set_index([('TNN HbA1c', 'last name', '')])
df.index.name = 'RIDER'
df.index = df.index.astype(int)
df = df.sort_index()

df = df.replace({'NV':np.nan})

df.to_csv(root+'HbA1c.csv')