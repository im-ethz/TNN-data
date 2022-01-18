import numpy as np
import pandas as pd
import datetime
from pytz import country_names, country_timezones

month_mapping = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july',
				 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
month_firstday = {'january':0, 'february':31, 'march':59, 'april':90, 'may':120, 'june':151, 'july':181,
				'august':212, 'september':243, 'october':273, 'november':304, 'december':334}

countries_eu = ('Albania', 'Andorra', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria',
				'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Faroe Islands', 'Finland',
				'France', 'Germany', 'Gibraltar', 'Greece', 'Guernsey', 'Hungary', 'Ireland', 
				'Isle of Man', 'Italy', 'Jersey', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 
				'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 
				'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 
				'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 
				'United Kingdom', 'Vatican City')

country_names_inv = {v:k for k,v in country_names.items()}
country_replace = {'Britain (UK)':'United Kingdom', 
					'Korea (South)':'South Korea', 
					'Korea (North)':'North Korea', 
					'Czech Republic':'Czechia',
					'Bosnia & Herzegovina':'Bosnia and Herzegovina'}
for old, new in country_replace.items():
	country_names_inv[new] = country_names_inv.pop(old)
country_add = {'Luzon':'PH'}
country_names_inv.update(country_add)

def create_calendar_array(df, datacol):
	"""
	Create array in the form of calendar (month x days)
	df 		- pandas DataFrame with dates as index (for only one year!)
	"""
	df['month'] = pd.to_datetime(df.index).month
	df['day'] = pd.to_datetime(df.index).day
	df_calendar = df.pivot('month','day', datacol)
	df_calendar.index = pd.CategoricalIndex(df_calendar.index.map(month_mapping), 
		ordered=True, categories=month_mapping.values())
	return df_calendar

def print_times_dates(text, df:pd.DataFrame, df_mask, ts='timestamp', verbose=False):
	print("\n", text)
	print("times: ", df_mask.sum())
	print("days: ", len(df[df_mask][ts].dt.date.unique()))
	if verbose:
		print("file ids: ", df[df_mask].file_id.unique())

def isnan(x):
	return (x != x)