# TODO: put features into different modalities
# TODO: clean out extreme values in script before
import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

from config import rider_mapping
rider_mapping_inv = {v:k for k, v in rider_mapping.items()}

from plot import *
from helper import *
from calc import calc_hr_zones, calc_power_zones
from calc import agg_power, agg_zones, agg_stats
from calc import chronic_training_load, acute_training_load, training_stress_balance

import gc

path = './data/'

# read in fitness variables
fitness = pd.read_csv(path+'fitness.csv', index_col=[0,1], header=[0,1])

# determine functional threshold power
FTP = fitness[[("VT2 (RCP)", 'W')]].dropna().reset_index().groupby('RIDER').mean()
LTHR = fitness[[("VT2 (RCP)", 'HR')]].loc[pd.IndexSlice[:, 'Jan_2019'],:].reset_index().drop('date', axis=1).set_index('RIDER')

# calculate HR and Power zones
hr_zones = LTHR.apply(calc_hr_zones, axis=1)
power_zones = FTP.apply(calc_power_zones, axis=1)

df_agg_day = {}

athletes = set(sorted([int(i.rstrip('.csv').rstrip('_info').rstrip('_data')) for i in os.listdir(path+'trainingpeaks/')]))

for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(path+'trainingpeaks/'+str(i)+'_data.csv', index_col=0)
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df['date'] = df['local_timestamp'].dt.date

	# calculate time statistics
	df_times = df.groupby('date').agg({'timestamp'			: ['min', 'max', 'count'],
									   'local_timestamp'	: ['min', 'max']})
	df_times.columns = [c1+'_'+c2 for c1, c2 in df_times.columns]

	# split out columns in ascent and descent
	df['descent'] = df.groupby('file_id')['altitude'].transform(lambda x: x.interpolate(method='linear').diff() < 0)
	for col in ['altitude', 'speed', 'distance', 'heart_rate', 'power', 'cadence']:
		df[col+'_up'] = df.loc[~df['descent'], col]
		df[col+'_down'] = df.loc[df['descent'], col]
	df.drop('descent', axis=1, inplace=True)

	# calculate features with removing zeros
	for col in ['power', 'combined_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness',
				'cadence', 'left_pedal_smoothness', 'right_pedal_smoothness']:
		try:
			df[col+'_n0'] = df[col].replace({0:np.nan})
		except KeyError:
			pass

	# calculate flirt statistics
	col_stats = set(df.columns)-set(['position_long', 'position_lat', 'gps_accuracy', 'battery_soc', 'device_0', 'local_timestamp', 'timestamp', 'time_training', 'file_id'])
	df_stats = df[col_stats].groupby('date').apply(agg_stats)
	df_stats.columns = [c1+'_'+c2 for c1, c2 in df_stats.columns]

	# clean columns
	df_stats.drop(df_stats.columns[df_stats.notna().sum() == 0], axis=1, inplace=True) #empty cols

	# calculate power statistics and hr and power zones
	df.set_index('timestamp', inplace=True)

	df_zones = df.groupby('date').apply(agg_zones, hr_zones=hr_zones.loc[i], power_zones=power_zones.loc[i])
	df_power = df.groupby('date').apply(agg_power, FTP=FTP.loc[i].item())

	df_power['chronic_training_load'] = chronic_training_load(df_power['training_stress_score'])
	df_power['acute_training_load'] = acute_training_load(df_power['training_stress_score'])
	df_power['training_stress_balance'] = training_stress_balance(df_power['chronic_training_load'], df_power['acute_training_load'])

	# read calories from info file
	df_files = df[['file_id', 'date']].drop_duplicates().set_index('file_id')
	df_info = pd.read_csv(path+'trainingpeaks/'+str(i)+'_info.csv', index_col=0)
	df_calos = df_files.join(df_info['total_calories']).groupby('date').sum()

	df_agg_day[i] = pd.concat([df_times, df_zones, df_power, df_calos, df_stats], axis=1)

	del df, df_times, df_zones, df_power, df_calos, df_stats, df_files, df_info

df_agg_day = pd.concat(df_agg_day)
df_agg_day.to_csv(path+'trainingpeaks_day.csv')