import numpy as np 
import pandas as pd 

# read data from FAO (2017) downloaded from https://ourworldindata.org/diet-compositions
df = pd.read_csv('./daily-caloric-supply-derived-from-carbohydrates-protein-and-fat.csv')

# select year 2013
df = df[df.Year == 2013]
df.reset_index(drop=True, inplace=True)
df.drop('Year', axis=1, inplace=True)

df.rename(columns={ 'Calories from animal protein (FAO (2017))'	:'animal protein (kcal)',
					'Calories from plant protein (FAO (2017))'	:'plant protein (kcal)',
					'Calories from fat (FAO (2017))'			:'fat (kcal)',
					'Calories from carbohydrates (FAO (2017))'	:'carbohydrates (kcal)'}, inplace=True)
df.to_csv('country_nutrients.csv', index_label=False)