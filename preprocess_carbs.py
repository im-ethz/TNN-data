import pandas as pd 

path = './data/carbs/'

# read data from FAO (2017) downloaded from https://ourworldindata.org/diet-compositions
df = pd.read_csv(path+'/daily-caloric-supply-derived-from-carbohydrates-protein-and-fat.csv')

# select most recent year (2013)
df = df[df.Year == df.Year.max()]
df = df.reset_index(drop=True)
df = df.drop('Year', axis=1)

df = df.rename(columns={'Entity'                                    :'country',
                        'Code'                                      :'code',
                        'Calories from animal protein (FAO (2017))'	:'animal protein (kcal)',
					    'Calories from plant protein (FAO (2017))'	:'plant protein (kcal)',
					    'Calories from fat (FAO (2017))'			:'fat (kcal)',
					    'Calories from carbohydrates (FAO (2017))'	:'carbohydrates (kcal)'})
df.to_csv(path+'/country_nutrients.csv', index_label=False)