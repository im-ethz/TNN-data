import pandas as pd
import numpy as np

path = './'

df = pd.read_excel(path+'2019_RaceCalendar_createdEva_checkKristina.xls', index_col=[0,1,2,3,4], header=[0], skipfooter=13)
df.reset_index(inplace=True)
df.rename(columns=df.iloc[0][:5].to_dict(), inplace=True)
df.drop(0, inplace=True)

df.drop(['name', 'country'], axis=1, inplace=True)

df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

df.replace({'Y':1}, inplace=True)

df.reset_index(inplace=True)
df.set_index(['index', 'start_date', 'end_date', 'type'], inplace=True)

df = df.astype(bool)

df.to_csv('racecalendar_2019_anonymous.csv')

df_list = df.T.stack().stack().stack().stack().reset_index()
df_list.rename(columns={'level_0':'RIDER'}, inplace=True)
df_list.sort_values(['RIDER', 'index'], inplace=True)

df_list = df_list[df_list[0]]
df_list.drop(0, axis=1, inplace=True)

df_list = df_list[['RIDER', 'start_date', 'end_date', 'type']]
df_list.reset_index(drop=True, inplace=True)
df_list.to_csv('racecalendar_2019_anonymous_list.csv')