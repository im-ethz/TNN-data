import pandas as pd
from pytz import country_names, country_timezones

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

def print_times_dates(text, df:pd.DataFrame, df_mask, ts='timestamp', verbose=False):
	print("\n", text)
	print("times: ", df_mask.sum())
	print("days: ", len(df[df_mask][ts].dt.date.unique()))
	if verbose:
		print("file ids: ", df[df_mask].file_id.unique())

def isnan(x):
	return (x != x)