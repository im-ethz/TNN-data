import os
import gc
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
#from joblib import Parallel, delayed

import datetime
import geopy
import pytz
from tzwhere import tzwhere
tzwhere = tzwhere.tzwhere()

from scipy.stats import zscore

from bike2csv.converter import Converter as Convert2CSV

from config import rider_mapping_inv, DATA_PATH
from helper import isnan, print_times_dates, country_names, country_timezones, country_names_inv
from calc import semicircles_to_degrees
from timezone import get_timezones_trainingpeaks

root = DATA_PATH+'TrainingPeaks/'

def get_country_from_gps(df:pd.DataFrame):
	# use geopy with OpenStreetMap to look up country from coordinates
	if 'position_lat' in df and 'position_long' in df:
		if not df['position_lat'].isna().all() and not df['position_long'].isna().all():
			geo = geopy.geocoders.Nominatim(user_agent="trainingpeaks_locations")
			return geo.reverse(query=str(df['position_lat'].dropna().iloc[0])+", "+str(df['position_long'].dropna().iloc[0]),
					language='en', zoom=0).raw['address']['country']
		else:
			return np.nan
	else:
		return np.nan

def get_timezone_from_location(df:pd.DataFrame):
	tz_loc_name = None
	if not df['country'].isna().all() and not (df['country'] == 'None').all():
		country_code = country_names_inv[df['country'].iloc[0]]
		if country_code is not None:
			country_tz = country_timezones[country_code]
			if len(country_tz) == 1:
				tz_loc_name = country_tz[0]
	
	if tz_loc_name is None:
		if 'position_lat' in df and 'position_long' in df:
			if not df['position_lat'].isna().all() and not df['position_long'].isna().all():
				tz_loc_name = tzwhere.tzNameAt(df['position_lat'].dropna().iloc[0], df['position_long'].dropna().iloc[0])
		
	if tz_loc_name is not None:
		return df['timestamp'] + pytz.timezone(tz_loc_name).utcoffset(df['timestamp'][0])
	else:
		return np.nan

def cleaning_per_session(df_data, df_info, j):
	# rename columns from deviating files (or remove them if they are duplicates)
	cols_rename = {'lat_gps'					:'position_lat',
					'lon_gps'					:'position_long',
					'pedal_smoothness_left' 	:'left_pedal_smoothness',
					'pedal_smoothness_right'	:'right_pedal_smoothness',
					'torque_effectiveness_left'	:'left_torque_effectiveness',
					'torque_effectiveness_right':'right_torque_effectiveness',
					'total_distance'			:'distance',
					'pedal_smoothness_combined'	:'combined_pedal_smoothness'}
	# remove duplicate columns if unknown column name
	if df_data.columns.isin(cols_rename.keys()).any():
		cols_dupl = df_data.columns[df_data.T.duplicated(keep=False)]
		df_data = df_data.drop(cols_dupl[cols_dupl.isin(cols_rename.keys())], axis=1) # drop duplicate columns
		if ('position_lat' in df_data and 'lat_gps' in df_data): # if they are not duplicates
			df_data = df_data.drop('lat_gps', axis=1)
		if ('position_long' in df_data and 'lon_gps' in df_data): # if they are not duplicates
			df_data = df_data.drop('lon_gps', axis=1)
		df_data = df_data.rename(columns=cols_rename) # rename any non-duplicate columns

	# TODO: we haven't covered the case where lat_gps is different from position_lat

	# convert latitude and longitude from semicircles to degrees
	if 'position_long' in df_data:
		df_data.loc[df_data['position_long'] == 0, 'position_long'] = np.nan
		df_data['position_long'] = semicircles_to_degrees(df_data['position_long'])
	if 'position_lat' in df_data:
		df_data.loc[df_data['position_lat'] == 0, 'position_lat'] = np.nan
		df_data['position_lat'] = semicircles_to_degrees(df_data['position_lat'])

	# get local timestamp
	df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
	# remove tz_info in timestamp (often incorrect, only occurs in garmin FIT-files from 2014)
	if df_data['timestamp'].dt.tz is not None:
		df_data['timestamp'] = df_data['timestamp'].dt.tz_convert(0).dt.tz_localize(None)

	try:
		# calculate timezone
		tz = (pd.to_datetime(df_info.loc['activity'].loc['local_timestamp'])
			- pd.to_datetime(df_info.loc['activity'].loc['timestamp']))[0]
		if abs(tz) > datetime.timedelta(days=1): # sometimes the local_timestamp is someday in 1989, so now I ignored it
			df_data['local_timestamp'] = np.nan
		else:
			df_data['local_timestamp'] = df_data['timestamp'] + tz
	except KeyError:
		df_data['local_timestamp'] = np.nan
	except TypeError: #tz is nan
		df_data['local_timestamp'] = np.nan

	df_data['country'] = get_country_from_gps(df_data)
	df_data['local_timestamp_loc'] = get_timezone_from_location(df_data)

	# fix duplicated entries in df_info from ELEMNT RIVAL devices
	df_info = df_info.reset_index().drop_duplicates().set_index(['level_0', 'level_1']) # note there is some bug in pandas that ignores the multi-index in drop duplicates
	df_info = df_info[~df_info.index.duplicated()] # remove device charge info (already in _device)
	df_info.index.names = (None, None) # remove index names

	# remove local_timestamp_loc if device is zwift
	if ('device_0', 'manufacturer') in df_info.T:
		if df_info.loc[('device_0', 'manufacturer')][0] == 'zwift':
			df_data['local_timestamp_loc'] = np.nan
			df_data['country'] = np.nan
	elif ('session', 'sub_sport') in df_info.T:
		if df_info.loc[('session', 'sub_sport')][0] == 'virtual_activity':
			df_data['local_timestamp_loc'] = np.nan
			df_data['country'] = np.nan

	# create column filename
	df_data['file_id'] = j

	return df_data, df_info

def product_info(x, col0):
	try:
		manufacturer = x[(col0, 'manufacturer')]
	except KeyError:
		manufacturer = np.nan
	try:
		product_name = x[(col0, 'product_name')]
	except KeyError:
		product_name = np.nan
	if isnan(manufacturer) and isnan(product_name):
		return np.nan
	else:
		return str(manufacturer) + ' ' + str(product_name)

def merge_sessions(i):
	if not os.path.exists(f'{root}clean/{i}'):
		os.makedirs(f'{root}clean/{i}')

	# merge individual training sessions per athlete
	df = []
	df_INFO = []

	# -------------------- Normal
	files = sorted(os.listdir(f'{root}csv/{i}/record'))
	for j, f in tqdm(enumerate(files)):
		df_data = pd.read_csv(f'{root}csv/{i}/record/{f}', index_col=0)
		df_info = pd.read_csv(f'{root}csv/{i}/info/{f[:-11]}_info.csv', index_col=(0,1))

		df_data, df_info = cleaning_per_session(df_data, df_info, j)

		df.append(df_data) ; del df_data

		# create df_information file
		df_info = df_info.T.rename(index={'0':j})
		df_INFO.append(df_info) ; del df_info

	df = pd.concat(df)
	df_INFO = pd.concat(df_INFO)

	# remove unknown columns
	df = df.drop(df.columns[df.columns.str.startswith('unknown')], axis=1)
	df.to_csv(f'{root}clean/{i}/{i}_data0.csv', index_label=False)

	# drop columns that only have zeros in them
	df_INFO = df_INFO.drop(df_INFO.columns[((df_INFO == 0) | df_INFO.isna()).all()], axis=1)

	# drop columns that start with unknown
	df_INFO = df_INFO.loc[:,~df_INFO.columns.get_level_values(1).str.startswith('unknown')]

	# combine all devices into a list
	cols_device = [c for c in df_INFO.columns.get_level_values(0).unique() if c[:6] == 'device']
	df_INFO[('device_summary', '0')] = df_INFO.apply(lambda x: product_info(x, 'device_0'), axis=1)
	df_INFO[('device_summary', '1')] = df_INFO.apply(lambda x: sorted([product_info(x, col0) for col0 in cols_device[1:]\
													if not isnan(product_info(x, col0))]), axis=1)
	df_INFO.to_csv(f'{root}clean/{i}/{i}_info.csv')

def clean(i, verbose=False):
	df = pd.read_csv(f'{root}clean/{i}/{i}_data0.csv', index_col=0)
	df_info = pd.read_csv(f'{root}clean/{i}/{i}_info.csv', header=[0,1], index_col=0)

	print("\n--------------- NAN")
	cols_ignore = set(['timestamp', 'local_timestamp', 'local_timestamp_loc', 'file_id', 'country'])

	# drop completely empty rows (usually first or last rows of a file)
	rows_nan = df.drop(cols_ignore, axis=1).isna().all(axis=1)
	df = df.dropna(how='all', subset=set(df.columns)-cols_ignore)
	print("DROP: %s rows due to being empty"%rows_nan.sum())

	print("\n--------------- DEVICE")
	# include device in df
	df = pd.merge(df, df_info[('device_summary', '0')].str.strip("nan").str.strip().rename('device_0'),
					left_on='file_id', right_index=True, how='left')

	df = df.replace({'device_0':{'wahoo_fitness ELEMNT'			:'ELEMNT',
								'wahoo_fitness ELEMNT BOLT'		:'ELEMNTBOLT',
								'wahoo_fitness ELEMNT ROAM'		:'ELEMNTROAM',
								'wahoo_fitness ELEMNT RIVAL'	:'ELEMNTRIVAL',
								'wahoo_fitness FITNESS'			:'WAHOO',
								'garmin'						:'GARMIN',
								'Garmin'						:'GARMIN',
								'Garmin Garmin Edge 510'		:'GARMIN',
								'Garmin Garmin Forerunner 910XT':'GARMINFORERUNNER',
								'Garmin fenix 3'				:'GARMINFENIX',
								'Garmin Unknow'					:'GARMINUNK',
								'Garmin Fietse'					:'GARMINUNK',
								'Garmin Hardlope'				:'GARMINUNK',
								'zwift'							:'ZWIFT',
								'bkool BKOOL Website'			:'BKOOL',
								'virtualtraining Rouvy'			:'ROUVY',
								'hammerhead Karoo'				:'KAROO',
								'pioneer'						:'PIONEER',
								'CycleOps'						:'CYCLEOPS',
								'Bryton'						:'BRYTON',
								'Golden Cheetah'				:'GOLDENCHEETAH',
								'strava'						:'STRAVA',
								'development'					:'DEVELOPMENT'}})
	df['device_0'] = df['device_0'].fillna('UNK')
	print("Devices used:\n", df.groupby('file_id').first().device_0.value_counts())

	# sort by device and timestamp for duplicates later
	df = df.sort_values(by=['device_0', 'timestamp'], 
		key=lambda x: x.map({'ELEMNTBOLT':0, 'ELEMENTROAM':1, 'ELEMNT':2, 'WAHOO':3, 'PIONEER':4, 'ZWIFT':5, 
			'GARMIN':6, 'ROUVY':7, 'BKOOL':8, 'ELEMENTRIVAL':9, 'GARMINFORERUNNER':10, 'GARMINFENIX':11, 'GARMINUNK':12,
			'KAROO':13, 'CYCLEOPS':14, 'BRYTON':15, 'STRAVA':16, 'DEVELOPMENT':17, 'GOLDENCHEETAH':18, 'UNK':19})) 
	df = df.reset_index(drop=True)

	devices = set(df['device_0'].unique())

	print("\n--------------- TIMESTAMP")
	# fix local timestamp
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df['local_timestamp_loc'] = pd.to_datetime(df['local_timestamp_loc'])

	if verbose:
		# print nans local_timestamp and local_timestamp_loc
		print("\n-------- Local timestamp: nan")
		print_times_dates("local timestamp not available", 
			df, df['local_timestamp'].isna())
		print_times_dates("local timestamp from location not available", 
			df, df['local_timestamp_loc'].isna())
		print_times_dates("local timestamp OR local timestamp from location not available", 
			df, df['local_timestamp'].isna() | df['local_timestamp_loc'].isna())
		print_times_dates("local timestamp AND local timestamp from location not available", 
			df, df['local_timestamp'].isna() & df['local_timestamp_loc'].isna())
	
	try:
		device_settings = df_info['device_settings'][['time_offset', 'time_zone_offset', 'utc_offset']]
		if ((device_settings != 0) & (device_settings.notna())).sum().sum() == 0:
			print("WARNING: time offset in device settings not zero")
	except KeyError:
		pass

	print("\n-------- Local timestamp: error")
	# print how often local_timestamp does not equal local_timestamp_loc
	# Note: this is mainly around the time that the clock is switched from summertime to wintertime. Find out what to do with it!
	nan_ts = df['local_timestamp'].isna() | df['local_timestamp_loc'].isna()
	print_times_dates("local timestamp does not equal local timestamp from location (excl. nans)", 
		df[~nan_ts], df[~nan_ts]['local_timestamp'] != df[~nan_ts]['local_timestamp_loc'])
	print("Dates for which this happens: ", 
		df[~nan_ts][df[~nan_ts]['local_timestamp'] != df[~nan_ts]['local_timestamp_loc']].timestamp.dt.date.unique())

	print("\n--------------- DUPLICATES")
	print("\n-------- Entire row duplicate")
	dupl = df.duplicated(set(df.columns)-set(['file_id']), keep=False)

	# check if one of the files continues on, or if it's simply a complete file that we drop
	len_dupl = {f :[len(df[df.file_id == f]), len(df[dupl & (df.file_id == f)])] for f in df[dupl].file_id.unique()}
	dupl_entire_file = dict(zip(len_dupl, [l==m for _, [l,m] in len_dupl.items()]))
	print("CHECK if we remove entire file by removing duplicated entries: ", dupl_entire_file)

	print("DROP: %s duplicate entries"%df.duplicated(set(df.columns)-set(['file_id'])).sum())
	df = df.drop_duplicates(subset=set(df.columns)-set(['file_id']), keep='first')

	print("\n-------- Timestamp duplicate")
	# Note: what can happen here is that two devices are recording at the same time for some reason
	# but one of the two devices may not contain all the information.
	print_times_dates("duplicated timestamps", 
		df[df['timestamp'].notna()], df[df['timestamp'].notna()]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df, df['local_timestamp'].duplicated(), ts='local_timestamp')

	# check if duplicate timestamps are caused by using multiple devices at the same time
	print("Duplicate timestamps in devices:")
	for dev in devices:
		print_times_dates(dev, df, df['timestamp'].duplicated(keep=False) & (df.device_0 == dev), ts='timestamp')

	# Due to the ordering of the data by device, the data from less desired devices is dropped
	# Copy duplicate timestamps to a separate files, in case we want to use it for imputation
	dupl = df[df['timestamp'].duplicated(keep='first')]
	dupl.to_csv(f'{root}clean/{i}/{i}_data_dupl.csv', index_label=False)
	df = df.drop_duplicates(subset='timestamp', keep='first')
	print("DROP: %s duplicate timestamps"%len(dupl))

	# save df to file
	df.to_csv(f'{root}clean/{i}/{i}_data1.csv', index_label=False)
	del df ; gc.collect()

def local_timestamp(i, verbose=False):
	################ PREREQUISITE: 
	## preprocess_dexcom.py: create {root}/timezone.csv
	tz = pd.read_csv(DATA_PATH+'/timezone.csv', index_col=0)
	tz['date'] = pd.to_datetime(tz['date'])
	tz['timezone'] = pd.to_timedelta(tz['timezone'])

	df = pd.read_csv(f'{root}clean/{i}/{i}_data1.csv')
	df['timestamp'] = pd.to_datetime(df['timestamp'])

	print("\n--------------- LOCAL TIMESTAMP")
	df = df.drop(['local_timestamp', 'local_timestamp_loc'], axis=1)
	df = df.sort_values('timestamp')

	df['date'] = pd.to_datetime(df['timestamp'].dt.date)

	df = pd.merge(df, tz.loc[tz['RIDER'] == i,  ['date', 'timezone']], on='date', how='left')
	df['local_timestamp'] = df['timestamp'] + df['timezone']
	df = df.drop(['date', 'timezone'], axis=1)

	df.to_csv(f'{root}clean/{i}/{i}_data2.csv', index_label=False)

def glucose():
	################ PREREQUISITE: 
	## preprocess_dexcom.py: create {root}/Dexcom/dexcom_clean.csv

	df_dc = pd.read_csv(DATA_PATH+'/Dexcom/clean/dexcom_clean.csv', index_col=0)
	df_dc['timestamp'] = pd.to_datetime(df_dc['timestamp'])
	df_dc['local_timestamp'] = pd.to_datetime(df_dc['local_timestamp'])

	print("\n--------------- GLUCOSE")

	athletes = sorted([int(i) for i in os.listdir(root+'clean/')])
	for i in athletes:
		print("\n----------- Athlete ", i)
		df = pd.read_csv(f'{root}clean/{i}/{i}_data2.csv')
		df['timestamp'] = pd.to_datetime(df['timestamp'])
		df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

		cols_glucose = ['glucose', 'Bloodglucose', 'Nightscout', 'Basal', 'InsulinOnBoard', 'CarbonBoard']
		cols_glucose = [col for col in cols_glucose if col in df]

		if len(cols_glucose) != 0: # so far always glucose in df (so don't catch exception)
			print("EXPORT glucose to dexcom")
			# get glucose out of trainingpeaks files
			df_glucose = df.dropna(subset=cols_glucose, how='all')
			df_glucose = df_glucose[['timestamp', 'local_timestamp'] + cols_glucose]

			# add info
			df_glucose['RIDER'] = i
			df_glucose['Event Type'] = 'EGV'
			df_glucose['source'] = 'TrainingPeaks'

			# merge all glucose columns
			for col in ['Bloodglucose', 'Nightscout']:
				if col in df_glucose:
					df_glucose['glucose'] = df_glucose['glucose'].fillna(df_glucose[col])
					df_glucose = df_glucose.drop(col, axis=1)

			for col in ['Basal', 'InsulinOnBoard', 'CarbonBoard']:
				if col in df_glucose:
					if df_glucose[col].notna().sum() != 0:
						raise NotImplementedError
					else:
						df_glucose = df_glucose.drop(col, axis=1)

			assert (df_glucose['glucose'] < 30).sum() == 0 # ensure that glucose units are mg/dl
			df_glucose = df_glucose.rename(columns={'glucose':'Glucose Value (mg/dL)'})

			df_dc = df_dc.append(df_glucose)

			df = df.drop(cols_glucose, axis=1)

		df.to_csv(f'{root}clean/{i}/{i}_data3.csv', index_label=False)
		del df, df_glucose; gc.collect()

	df_dc = df_dc.sort_values(['RIDER', 'timestamp'])
	df_dc.to_csv(DATA_PATH+'/Dexcom/clean/dexcom_clean2.csv', index_label=0)

def features(i, verbose=0):
	df = pd.read_csv(f'{root}clean/{i}/{i}_data3.csv')
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	# TODO: do this step earlier
	# filter out weird timestamps in 2024 and 2007(that are clearly incorrect)
	# (happens only once for athlete 14 and twice for athlete 8)
	df = df[df.timestamp <= '2021-12-31 23:59:59']
	df = df[df.timestamp >= '2014-01-01 00:00:00']

	"""
	print("\n-------- select devices")
	# select devices
	# because there is a high percentage of missing values for heart rate
	# we cannot entirely trust the timestamps on it, we've seen weird things when people use the garmin regarding to the timestamp
	# therefore better to drop it entirely
	# note that we do drop ~ roughly 10% of the data in this step
	devices = set(df.device_0.unique())
	keep_devices = set(['ELEMNT', 'ELEMNTBOLT', 'ELEMNTROAM', 'ZWIFT'])

	print("DROP: %s files with %s entries from devices %s"\
		%(len(df[~df['device_0'].isin(keep_devices)].file_id.unique()), (~df['device_0'].isin(keep_devices)).sum(), devices - keep_devices))
	df = df[df['device_0'].isin(keep_devices)]
	"""

	print("\n-------- FEATURES")
	# -------------------- Rename
	# TODO: fix this in earlier stage
	cols_rename = {'total_distance'				:'distance',
				   'pedal_smoothness_combined'	:'combined_pedal_smoothness'}
	for col_from, col_to in cols_rename.items():
		if col_from in df:
			df[col_to] = df[col_to].fillna(df[col_from])
			df = df.drop(col_from, axis=1)
			print(f"FILLNA {col_to} with {col_from}")

	# -------------------- Empty
	cols_empty = ('cadence256', 'speed_1s', 'time128', 'vertical_oscillation', 'total_cycles', 
		'cycles', 'stance_time_percent', 'stance_time', 'ball_speed', 'stroke_type', 'zone',
		'device_index', 'total_hemoglobin_conc', 'total_hemoglobin_conc_min', 
		'total_hemoglobin_conc_max', 'saturated_hemoglobin_percent', 
		'saturated_hemoglobin_percent_min', 'saturated_hemoglobin_percent_max')
	for col in cols_empty: #TODO: don't do this iteratively
		if col in df:
			df = df.drop(col, axis=1)
			print("DROP: ", col)

	# -------------------- Irrelevant and sparse columns
	print("Percentage of missing values for each features:\n",
		df.isna().sum() / len(df))
	print("Columns with more than 90% missing:\n", 
		df.columns[df.isna().sum() / len(df) > 0.9])
	# drop irrelevant features
	# note: pwrright is only there when power is there, so no additional info
	# note: vertical_speed is only there when grade is there, so no additional info
	cols_drop = ('gps_accuracy', 'battery_soc', 'calories', 'accumulated_power', 
				'fractional_cadence', 'time_from_course', 'compressed_speed_distance', 
				'resistance', 'cycle_length', 'pwrright', 'vertical speed',
				'activity_type', 'SensorState', 'torq')
	for col in cols_drop: # TODO: don't do this iteratively
		if col in df:
			df = df.drop(col, axis=1)
			print("DROP: ", col)

	# TODO: find out how activity type ended up here
	# TODO 3, 12, 13 negative distance diffs

	# -------------------- Time in session
	# create column time in session
	print("CREATE: time session")
	df['time_session'] = df.groupby('file_id')['timestamp'].apply(lambda x: x - x.min()) / np.timedelta64(1,'s')

	if verbose > 0:
		# length cycling
		duration_session = df.groupby('file_id').count().max(axis=1) / 60
		#PlotPreprocess(f'{root}/clean/{i}/', athlete=i).plot_hist(duration_session, 'duration_session (min)')
		print("Max training length (min): ", duration_session.max())
		print("Number of cycling sessions that last shorter than 10 min: ", (duration_session <= 10).sum())
		print("Number of cycling sessions that last shorter than 20 min: ", (duration_session <= 20).sum())
		del duration_session

	# -------------------- Distance
	# when position_lat and position_long are missing, set value of distance to nan
	df.loc[df['position_lat'].isna(), 'distance'] = np.nan

	# clean distance > 1000km (note this works for now, but in the future we should reset it and add it up again)
	df.loc[df['distance'] > 1e6, 'distance'] = np.nan	
	print("CLEAN: distance")

	""" TODO
	# calculate speed from distance
	# interpolate for nans - then average over timestamps (in case timestamps are missing) - then take rolling mean over 5 sec
	for fid in df.file_id.unique():
		df.loc[df.file_id == fid, 'speed2'] = (df.loc[df.file_id == fid, 'distance'].interpolate(method='linear').diff() / df.loc[df.file_id == fid, 'local_timestamp'].diff().dt.seconds).rolling(5).mean()

	# remove measurements with speed higher than 200 km/h
	#df.loc[df['speed2'] > 200*1000/3600, 'distance'] = np.nan
	df = df.drop('speed2', axis=1)
	"""

	# -------------------- Left-Right Balance
	# there are some strings in this column for some reason (e.g. 'mask', 'right')
	if df.left_right_balance.apply(lambda x: isinstance(x, str)).any():
		df.left_right_balance = df.left_right_balance.replace({'mask':np.nan, 'right':np.nan})
	df.left_right_balance = pd.to_numeric(df.left_right_balance)
	print("CLEAN: left-right balance")

	# -------------------- Heart rate
	if df.heart_rate.apply(lambda x: isinstance(x, str)).any():
		df['heart_rate'] = df['heart_rate'].apply(lambda x: re.sub(' +', '', x) if isinstance(x, str) else x).replace({'\n':np.nan})
	df['heart_rate'] = pd.to_numeric(df['heart_rate'])
	print("CLEAN: heart_rate")

	# -------------------- Enhanced altitude
	# check if enhanced_altitude equals altitude
	print("CHECK: enhanced altitude does not equal altitude %s times"\
		%((df['enhanced_altitude'] != df['altitude']) & 
		(df['enhanced_altitude'].notna()) & (df['altitude'].notna())).sum())
	print("DROP: altitude (equals enhanced_altitude)")
	df = df.drop('altitude', axis=1)
	df = df.rename(columns={'enhanced_altitude':'altitude'})

	# -------------------- Enhanced speed
	# check if enhanced_speed equals speed
	print("CHECK: enhanced speed does not equal speed %s times"\
		%((df['enhanced_speed'] != df['speed']) & 
		(df['enhanced_speed'].notna()) & (df['speed'].notna())).sum())
	print("DROP: speed (equals enhanced_speed)")
	df = df.drop('speed', axis=1)
	df = df.rename(columns={'enhanced_speed':'speed'})

	# -------------------- Power
	z = zscore(df['power'], nan_policy='omit')
	z_extreme = (df.loc[np.abs(z) > 3, 'file_id'].value_counts() / df.groupby('file_id')['timestamp'].count()) > 0.7
	fid_extreme = z_extreme[z_extreme].index
	for fid in fid_extreme:
		df.loc[df.file_id == fid, 'power'] /= 10

	# -------------------- Elevation gain # TODO: do we use this later on?
	df['elevation_gain'] = df.groupby('file_id')['altitude'].apply(lambda x: x.interpolate(method='linear').diff())
	df.loc[df['timestamp'].diff() > '1s', 'elevation_gain'] = np.nan
	print("CREATE: elevation gain")
	# TODO: remove extreme values

	# -------------------- Acceleration # TODO: do we use this later on?
	df['acceleration'] = df.groupby('file_id')['speed'].apply(lambda x: x.interpolate(method='linear').diff())
	df.loc[df['timestamp'].diff() > '1s', 'acceleration'] = np.nan
	print("CREATE: acceleration")
	# TODO: remove extreme values

	# -------------------- Distance
	# negative distance diffs (meaning that the athletes is moving backwards)
	print("CHECK: Negative distance diffs: ", 
		((df.time_session.diff() == 1) & (df['distance'].diff() < 0)).sum())
	# only for athlete 10 there are two negative distance diffs

	df.to_csv(f'{root}clean/{i}/{i}_data4.csv', index_label=False)

	# TODO: combined_pedal_smoothness

def drop_extremes(df, feature, comp, value):
	if feature in df:
		if comp == 'lower':
			mask = df[feature] <= value
		elif comp == 'higher':
			mask = df[feature] >= value
		elif comp == 'equal':
			mask = df[feature] == value

		if mask.sum() > 0:
			print(f"DROP: {mask.sum()} with {feature} {comp} than {value}")
		df.loc[mask, feature] = np.nan
	return df

def extremes(i):
	df = pd.read_csv(f'{root}clean/{i}/{i}_data4.csv')

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	# heart rate
	df = drop_extremes(df, 'heart_rate', 'lower', 0)
	df = drop_extremes(df, 'heart_rate', 'higher', 250)

	# power 
	# note: it can happen that they do not deliver any power for a minute.
	# should it then be 0 or nan??
	df = drop_extremes(df, 'power', 'lower', 0)
	df = drop_extremes(df, 'power', 'higher', 3000)

	# temperature
	df = drop_extremes(df, 'temperature', 'higher', 60)
	df = drop_extremes(df, 'temperature', 'lower', -30)

	# altitude
	df = drop_extremes(df, 'altitude', 'lower', -500)
	df = drop_extremes(df, 'altitude', 'higher', 5000)

	# speed
	df = drop_extremes(df, 'speed', 'higher', 300/3.6)
	df = drop_extremes(df, 'speed', 'lower', -1e-10)

	# grade
	df = drop_extremes(df, 'grade', 'higher', 100)
	df = drop_extremes(df, 'grade', 'lower', -100)

	# cadence 
	df = drop_extremes(df, 'cadence', 'lower', 0)
	df = drop_extremes(df, 'cadence', 'higher', 300)

	# left_torque_effectivenss
	df = drop_extremes(df, 'left_torque_effectiveness', 'lower', -1)
	df = drop_extremes(df, 'left_torque_effectiveness', 'higher', 101)	

	# right_torque_effectivenss
	df = drop_extremes(df, 'right_torque_effectiveness', 'lower', -1)
	df = drop_extremes(df, 'right_torque_effectiveness', 'higher', 101)	

	# combined_pedal_smoothness
	df = drop_extremes(df, 'combined_pedal_smoothness', 'lower', -1)
	df = drop_extremes(df, 'combined_pedal_smoothness', 'higher', 101)	

	# left_pedal_smoothness
	df = drop_extremes(df, 'left_pedal_smoothness', 'lower', -1)
	df = drop_extremes(df, 'left_pedal_smoothness', 'higher', 101)	

	# right_pedal_smoothness
	df = drop_extremes(df, 'right_pedal_smoothness', 'lower', -1)
	df = drop_extremes(df, 'right_pedal_smoothness', 'higher', 101)	

	df.to_csv(f'{root}clean/{i}/{i}_data5.csv', index_label=False)

def get_session_times():
	athletes = sorted([int(i) for i in os.listdir(root+'csv/')])
	df_training = {}
	for i in athletes:
		print(i)
		df_i = pd.read_csv(f'{root}clean/{i}/{i}_data4.csv')
		df_i['timestamp'] = pd.to_datetime(df_i['timestamp'])
		df_training[i] = df_i.groupby('file_id').agg({'timestamp' : ['min', 'max']})
		del df_i ; gc.collect()

	df_training = pd.concat(df_training)
	df_training.columns = ['_'.join(col) for col in df_training.columns]
	df_training = df_training.reset_index().rename(columns={'level_0':'RIDER'})
	df_training.to_csv(root+'session_times.csv', index_label=False)

def main():
	athletes = sorted([int(i) for i in os.listdir(root+'csv/')])

	# merge all csv files
	for i in athletes:
		print("\n------------------------------- Athlete ", i)
		path = dict(_path_zip = os.path.join(root, 'export', rider_mapping_inv[i]),
					_path_fit = os.path.join(root, 'fit', str(i)),
					_path_csv = os.path.join(root, 'csv', str(i))) 

		converter = Convert2CSV(**path)

		for file in converter.files:
			try:
				converter.convert(file)
			except: # NotImplementedError (for now ignore)
				continue

		merge_sessions(i)

		clean(i) # after this we can continue running preprocess_dexcom.py (and timezone.py)

		get_timezones_trainingpeaks(df, i, root)

		local_timestamp(i)

	glucose()

	for i in athletes:
		print("\n------------------------------- Athlete ", i)
		features(i)
		extremes(i)

	get_session_times()
	# TODO: use timezone.csv to fillna countries and timezones

if __name__ == '__main__':
	main()