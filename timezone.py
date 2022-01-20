# Find out the timezone windows for each rider based on gaps and duplicates in the data.
# Then put it next to the travelling that we know from trainingpeaks, and check if all the travelling checks out.

# GAP
# There can be two causes of gaps: travelling and censoring
# By using the difference between the local_timestamp_diff and the transmitter_diff,
# we wish to correct for the censoring.

# DUP
# There can be only one cause of duplicates (besides some bug): travelling.

# MANUAL check of the right timezones

# OBSERVATIONS:
# - It can happen that they travel and not record at the same time.
# - The phone corrects the timezone automatically (we think?) 
#   and the dexcom recorder device (starting with PL8..) has to be reset manually.
# - The timezone updates of dexcom do not match with those of trainingpeaks at all,
#   meaning that there is often a delay of 1 day to 3 weeks. Therefore the trainingpeaks file
#   can be used only as roughly a guide of which timezones they changed to.
# - Most likely trainingpeaks contains the most accurate travelling information, but the riders just
#   don't change the timezone of the cgm immediately.

# CONCLUSION The local timestamp of dexcom is very often incorrect and does is not the actual local timestamp
# Therefore for any type of analysis, we have to use UTC and the timezone and calculate a new local timestamp.

import os

import numpy as np
import pandas as pd

import datetime

from helper import countries_eu

# Get dates on which daylight savings time changes for relevant countries
dst_change = {'Australia':( '2014-10-05', '2014-04-06',
							'2015-10-04', '2015-04-05',
							'2016-10-02', '2016-04-03',
							'2017-10-01', '2017-04-02',
							'2018-10-07', '2018-04-01',
							'2019-10-06', '2019-04-07',
							'2020-10-04', '2020-04-05',
							'2021-10-03', '2021-04-04'),
			  'United States':( '2014-03-09', '2014-11-02', 
								'2015-03-08', '2015-11-01',
								'2016-03-13', '2016-11-06',
								'2017-03-12', '2017-11-05',
								'2018-03-11', '2018-11-04',
								'2019-03-10', '2019-11-03',
								'2020-03-08', '2020-11-01',
								'2021-03-14', '2021-11-07',),
			  'Turkey': ('2014-03-31', '2014-10-26',
						 '2015-03-29', '2015-11-08',
						 '2016-03-27', '2016-09-08'),
			  'Mexico': ('2014-04-06', '2014-10-26',
						 '2015-04-05', '2015-10-25',
						 '2016-04-03', '2016-10-30',
						 '2017-04-02', '2017-10-29',
						 '2018-04-01', '2018-10-28',
						 '2019-04-07', '2019-10-27',
						 '2020-04-05', '2020-10-25',
						 '2021-04-04', '2021-10-31'),
			  'Brazil': ('2014-10-19', '2014-02-16',
						 '2015-10-18', '2015-02-22',
						 '2016-10-16', '2016-02-21',
						 '2017-10-15', '2017-02-19',
						 '2018-11-04', '2018-02-18',
						 '2017-02-17'),
			  'New Zealand': ('2014-09-28', '2014-04-06',
							  '2015-09-27', '2015-04-05',
							  '2016-09-25', '2016-04-03',
							  '2017-09-24', '2017-04-02',
							  '2018-09-30', '2018-04-01',
							  '2019-09-29', '2019-04-07',
							  '2020-09-27', '2020-04-05',
							  '2021-09-26', '2021-04-04')}
dst_change['Canada'] = dst_change['United States']
dst_change.update({c: ('2014-03-30', '2014-10-26', 
					'2015-03-29', '2015-10-25',
					'2016-03-27', '2016-10-30',
					'2017-03-26', '2017-10-29',
					'2018-03-25', '2018-10-28',
					'2019-03-31', '2019-10-27',
					'2020-03-29', '2020-10-25',
					'2021-03-28', '2021-10-31')for c in countries_eu})

############################# TrainingPeaks #############################
def get_timezones_trainingpeaks(df, i, root):
	print("\n--------------- TIMEZONES")
	df = df.sort_values('timestamp')

	tz = df[['timestamp', 'country', 'local_timestamp', 'local_timestamp_loc', 'file_id', 'device_0']]
	tz['timezone'] = tz['local_timestamp'] - tz['timestamp']
	tz['timezone_loc'] = tz['local_timestamp_loc'] - tz['timestamp']
	tz = tz.groupby(['file_id']).agg({	'timestamp'				: ['min', 'max'],
										'local_timestamp'		: ['min', 'max'],
										'local_timestamp_loc'	: ['min', 'max'],
										'device_0'				: 'first',
										'country'				: 'first',
										'timezone'				: 'first',
										'timezone_loc'			: 'first'})
	
	tz.columns = [i[0]+'_'+i[1] if 'timestamp' in i[0] else i[0] for i in tz.columns]
	tz = tz.sort_values('timestamp_min')

	tz['date'] = tz['timestamp_min'].dt.date
	tz = tz.reset_index()
	tz.to_csv(f'{root}clean/{i}/{i}_timezone_raw.csv')

	# fill nan timzones and country based on days before and after
	tz = fill_timezones(tz)
	tz.to_csv(f'{root}clean/{i}/{i}_timezone_filled.csv')

	# combine timezones
	tz = tz.drop('timezone', axis=1)
	tz = tz.rename(columns={'timezone_loc':'timezone'})
	tz.to_csv(f'{root}clean/{i}/{i}_timezone_combine.csv')

	tz = tz[['file_id', 'device_0', 'timestamp_min', 'timestamp_max', 'date', 'timezone', 'country']]

	tz['local_timestamp_min'] = tz['timestamp_min'] + tz['timezone']
	tz['local_timestamp_max'] = tz['timestamp_max'] + tz['timezone']
	tz.to_csv(f'{root}clean/{i}/{i}_timezone_final_list.csv')

	# drop unknown timezones
	tz = tz.dropna(subset=['timezone'])

	# TODO: check how it is possible that country is not nan, but timezone is
	# note: 3, 2030 and 3, 2031 are wrong
	# 6,82 and 6,117 are wrong (maybe even wrong date?)
	# 10,649 is wrong (probably even wrong date)
	# 12, 2240, 2242, 2244, 2246, 2248, 2250, 2253, 2255, 2260, 2262, 2263, 2265, 2267, 2269, 2273, 2275, 2277, 2280, until 2020-04-06

	# MANUAL drop timestamps that seem very odd (i.e. travelling from Japan to Spain and back in one day)
	# TODO: tz = tz.drop([1040,2827,2832])

	# create n_location column
	tz['n'] = (tz['timezone'].shift() != tz['timezone']).cumsum()

	tz = tz.groupby('n').agg({	'timezone'				:'first',
								'date'					:['min', 'max'], 
								'timestamp_min'			:'min',
								'timestamp_max'			:'max',
								'local_timestamp_min'	:'min',
								'local_timestamp_max'	:'max'})
	tz.columns = [i[0]+'_'+i[1] if i[0] == 'date' else i[0] for i in tz.columns]
	tz.to_csv(f'{root}clean/{i}/{i}_timezone_final.csv')


############################# Dexcom #############################
def calc_timezone_diff(df):

	df['local_timestamp_diff'] = df['local_timestamp'].diff()
	df['transmitter_diff'] = df['Transmitter Time (Long Integer)'].diff()

	# calculate difference between local_timestamp_diff and transmitter_diff to correct for travelling
	# but remember that when the transmitter changes, we cannot do this
	df['timediff'] = df['local_timestamp_diff'] - pd.to_timedelta(df['transmitter_diff'], 'sec')
	df.loc[df['transmitter_order'].diff() != 0, 'timediff'] = np.nan # correct for transmitter change

	return df

def calc_timezone_change(df):
	# Gap
	# Identify travelling gap: when local_timestamp_diff - transmitter_diff > 5min 
	# (note doesn't matter if it is 55 min or 5 min). 
	# We do a little bit more than 0 in case there are issues with the receiver 
	# (happens quite often that there is a lag.)
	# A gap from travelling should be at least 1 hour away.
	df['gap'] = (df['timediff'] > '5min')
	print("Number of gaps: ", df['gap'].sum())

	# Dup
	# Identify travelling dup: when local_timestamp_diff - transmitter_diff < -5 min
	# (note here it again doesn't matter if it is 55 min or 5 min). 
	# We do a little more than 0 in case there are issues with the receiver.
	df['dup'] = (df['timediff'] < '-5min')
	print("Number of dups: ", df['dup'].sum())

	df['change'] = df['dup'] | df['gap']
	print("Number of changes: ", df['change'].sum())

	return df

def get_timezone_changelist(df):

	# max local timestamp spend in the timezone
	df['local_timestamp_max'] = df['local_timestamp'].shift(1)
	df['index_max'] = df['index'].shift(1)

	# create list
	df_changes = df.loc[df['change'] | (df.RIDER.diff() != 0), 
		['index', 'index_max', 'RIDER', 'local_timestamp', 'local_timestamp_max', 'timediff']]

	# min local timestamp spend in the timezone
	df_changes = df_changes.rename(columns={'local_timestamp':'local_timestamp_min', 'index':'index_min'})

	# shift columns upwards
	df_changes['index_max'] = df_changes['index_max'].shift(-1)
	df_changes['local_timestamp_max'] = df_changes['local_timestamp_max'].shift(-1)
	df_changes['timediff'] = df_changes['timediff'].shift(-1)

	# number of timezone change
	df_changes['n'] = df_changes.groupby('RIDER').cumcount()

	# sort columns
	df_changes = df_changes[['RIDER', 'n', 'index_min', 'index_max', 'local_timestamp_min', 'local_timestamp_max', 'timediff']]

	df_changes = df_changes.set_index(['RIDER', 'n'])

	return df_changes

def get_transmitter_changelist(df):
	df_transmitter_change = df.loc[df.transmitter_order.diff() != 0, ['RIDER', 'index', 'index_max', 'local_timestamp', 'local_timestamp_max']]
	df_transmitter_change.rename(columns={'local_timestamp':'local_timestamp_min',
								'index':'index_min'}, inplace=True)
	df_transmitter_change['index_max'] = df_transmitter_change['index_max'].shift(-1)
	df_transmitter_change['local_timestamp_max'] = df_transmitter_change['local_timestamp_max'].shift(-1)# 4,9!!
	df_transmitter_change['n'] = df_transmitter_change.groupby('RIDER').cumcount()
	df_transmitter_change.set_index(['RIDER', 'n'], inplace=True)
	return df_transmitter_change

def insert_transmitter_change(df, idx_insert, df_tm, idx_tm, t_start):
	"""
	Insert rows of time changes that were not identified with the dexcom timezone changes
	e.g. can happen if an athlete changes transmitters
	
	Arguments:
		df 			dataframe with timezone changes
		df_tm 		dataframe with transmitter changes
		idx_insert	index in (RIDER, n) where to insert in df
		idx_tm		indices (tuple) of max date and min date in transmitter changes
		t_start		timediff to start with (find out manually)
	Returns
		dataframe with missing timezone change inserted
	"""
	i = df.index.get_loc(idx_insert)
	if isinstance(i, slice): # sometimes it randomly returns a slice
		i = i.start

	df_A = df.iloc[:i]

	rows = pd.DataFrame.from_records(data=(2)*[df.iloc[i]], 
		index=pd.MultiIndex.from_tuples((2)*[idx_insert], names=['RIDER', 'n']))

	rows['index_max'].iloc[0] = df_tm.loc[idx_tm[0], 'index_max']
	rows['index_min'].iloc[1] = df_tm.loc[idx_tm[1], 'index_min']

	rows['local_timestamp_max'].iloc[0] = df_tm.loc[idx_tm[0], 'local_timestamp_max']
	rows['local_timestamp_min'].iloc[1] = df_tm.loc[idx_tm[1], 'local_timestamp_min']

	rows['timediff'].iloc[0] = np.nan

	df_B = df.iloc[i+1:]

	df_A = df_A.append(rows).append(df_B)

	# calculate timediff
	df_A.loc[idx_insert, 'timediff'].iloc[0] = pd.to_timedelta(t_start)

	# reset counter n
	df_A = df_A.reset_index()
	df_A['n'] = df_A.groupby('RIDER').cumcount()
	df_A = df_A.set_index(['RIDER', 'n'])
	return df_A

def get_timezones_dexcom(df):
	"""
	Obtain timezone changes from the Dexcom data
	Note that these changes can be done manually and are thus often incorrect
	"""
	# calculate timezones only for EGV readings
	df_egv = df[df['Event Type'] == 'EGV']
	df_egv = df_egv.reset_index()

	# identify timezone windows for each rider
	df_egv = calc_timezone_diff(df_egv)
	df_egv = calc_timezone_change(df_egv)

	# create list of timezone changes
	df_changes = get_timezone_changelist(df_egv)

	# -------- Add missing timezone changes
	# sometimes we don't observe a timezone change when an athlete was not wearing a transmitter 
	# and was in between transmitters
	df_transmitter_change = get_transmitter_changelist(df_egv)
	df_changes = insert_transmitter_change(df_changes, (2,8), df_transmitter_change, ((2,5), (2,6)), '-7h')
	df_changes = insert_transmitter_change(df_changes, (3,7), df_transmitter_change, ((3,1), (3,2)), '2h')
	df_changes = insert_transmitter_change(df_changes, (3,17), df_transmitter_change, ((3,4), (3,5)), '6h')
	df_changes = insert_transmitter_change(df_changes, (3,24), df_transmitter_change, ((3,5), (3,6)), '-6h')
	df_changes = insert_transmitter_change(df_changes, (3,30), df_transmitter_change, ((3,7), (3,8)), '1h')
	df_changes = insert_transmitter_change(df_changes, (4,0), df_transmitter_change, ((4,1), (4,2)),'12h')
	df_changes = insert_transmitter_change(df_changes, (4,26), df_transmitter_change, ((4,6), (4,7)), '-10h')
	df_changes = insert_transmitter_change(df_changes, (5,0), df_transmitter_change, ((5,0), (5,1)), '1h')
	df_changes = insert_transmitter_change(df_changes, (5,1), df_transmitter_change, ((5,1), (5,2)), '-1h')
	df_changes = insert_transmitter_change(df_changes, (5,3), df_transmitter_change, ((5,2), (5,3)), '-1h')
	df_changes = insert_transmitter_change(df_changes, (6,46), df_transmitter_change, ((6,5), (6,6)), '-1h')
	df_changes = insert_transmitter_change(df_changes, (6,47), df_transmitter_change, ((6,6), (6,7)), '8h')
	df_changes = insert_transmitter_change(df_changes, (6,114), df_transmitter_change, ((6,9), (6,10)), '-1 days +15:55:00')
	df_changes = insert_transmitter_change(df_changes, (7,0), df_transmitter_change, ((7,0), (7,1)), '1h')
	df_changes = insert_transmitter_change(df_changes, (10,16), df_transmitter_change, ((10,1), (10,2)), '-7h')
	df_changes = insert_transmitter_change(df_changes, (10,333), df_transmitter_change, ((10,7), (10,8)), '0 days 00:56:00')
	df_changes = insert_transmitter_change(df_changes, (10,384), df_transmitter_change, ((10,11), (10,12)), '5h')
	df_changes = insert_transmitter_change(df_changes, (11,12), df_transmitter_change, ((11,4), (11,5)), '-6h')
	df_changes = insert_transmitter_change(df_changes, (11,26), df_transmitter_change, ((11,6), (11,7)), '-1h')
	df_changes = insert_transmitter_change(df_changes, (12,6), df_transmitter_change, ((12,0), (12,1)), '1h')
	df_changes = insert_transmitter_change(df_changes, (15,3), df_transmitter_change, ((15,0), (15,1)), '0 days 01:01:20')
	df_changes = insert_transmitter_change(df_changes, (15,4), df_transmitter_change, ((15,1), (15,2)), '6h')
	df_changes = insert_transmitter_change(df_changes, (15,5), df_transmitter_change, ((15,2), (15,3)), '-7h')
	df_changes = insert_transmitter_change(df_changes, (15,10), df_transmitter_change, ((15,3), (15,4)), '0 days 01:02:00')
	df_changes = insert_transmitter_change(df_changes, (15,15), df_transmitter_change, ((15,6), (15,7)), '0 days 01:02:00')
	df_changes = insert_transmitter_change(df_changes, (16,4), df_transmitter_change, ((16,1), (16,2)), '0 days 03:55:00')
	df_changes = insert_transmitter_change(df_changes, (17,2), df_transmitter_change, ((17,0), (17,1)), '-7h')
	df_changes = insert_transmitter_change(df_changes, (18,10), df_transmitter_change, ((18,1), (18,2)), '11min')
	df_changes = insert_transmitter_change(df_changes, (18,18), df_transmitter_change, ((18,2), (18,3)), '0 days 01:20:00')
	df_changes = insert_transmitter_change(df_changes, (18,29), df_transmitter_change, ((18,4), (18,5)), '-2h')
	df_changes = insert_transmitter_change(df_changes, (18,43), df_transmitter_change, ((18,8), (18,9)), '0 days 05:02:00')

	# -------- Initialize timezones
	# these are chosen in such a way that the final timezones overlap (roughly) 
	# with the actual timezone changes observed in trainingpeaks
	df_changes.loc[(1,0), 'timezone'] = pd.to_timedelta('-1 days +19:58:00')
	df_changes.loc[(2,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(3,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(4,0), 'timezone'] = pd.to_timedelta('-1 days +19:59:00')
	df_changes.loc[(5,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(6,0), 'timezone'] = pd.to_timedelta('-1 days +19:00:00')
	df_changes.loc[(7,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(9,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(10,0), 'timezone'] = pd.to_timedelta('-1 days +19:00:00')
	df_changes.loc[(11,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(12,0), 'timezone'] = pd.to_timedelta('4h')
	df_changes.loc[(13,0), 'timezone'] = pd.to_timedelta('0 days 00:56:21')
	df_changes.loc[(14,0), 'timezone'] = pd.to_timedelta('-1 days +23:55:00') # THIS IS A GUESS (no way to found out)
	df_changes.loc[(15,0), 'timezone'] = pd.to_timedelta('4h')
	df_changes.loc[(16,0), 'timezone'] = pd.to_timedelta('-1 days +19:58:00')
	df_changes.loc[(17,0), 'timezone'] = pd.to_timedelta('2h')
	df_changes.loc[(18,0), 'timezone'] = pd.to_timedelta('-1 days +19:00:00') #- pd.to_timedelta('-2 days +21:59:31')

	# -------- Calculate final timezones
	# calculate final timezone
	athletes = df_changes.index.get_level_values(0).unique()
	for i in athletes:
		for n in df_changes.loc[i].index[1:]:
			df_changes.loc[(i,n), 'timezone'] = df_changes.loc[(i,n-1), 'timezone'] + df_changes.loc[(i,n-1), 'timediff']

	# set last index and timestamp
	df_changes.loc[:,'index_max'].iloc[-1] = df.index[-1]
	df_changes.loc[:,'local_timestamp_max'].iloc[-1] = df.iloc[-1].local_timestamp

	df_changes.index_max = df_changes.index_max.astype(int)

	df_changes = df_changes.drop('timediff', axis=1)

	return df_changes

############################# TrainingPeaks + Dexcom #############################

def fill_timezones(tz, cols_fill=('timezone', 'timezone_loc', 'country')):
	# Fill up NaN columns from two days before and after
	for col in cols_fill:
		for idx, d in tz.loc[tz[col].isna(), 'date'].items():
			# take time window of two days before and after the date
			mask_prev = (tz.date >= d-datetime.timedelta(days=2)) & (tz.date <= d)
			mask_next = (tz.date <= d+datetime.timedelta(days=2)) & (tz.date >= d)

			col_prev = tz[mask_prev][col].dropna()
			col_next = tz[mask_next][col].dropna()

			# if there is at least one entry before and one entry after within the time window
			# and the "col" of these entries is the same
			if len(col_prev) >= 1 and len(col_next) >= 1 and len(pd.concat([col_prev, col_next]).unique()) == 1:
				tz.loc[idx, col] = pd.concat([col_prev, col_next]).unique()[0]

	# Fill up NaN columns from file before and after
	for col in cols_fill:
		for idx in tz.loc[tz[col].isna()].index:
			# take time window of file before and file after with not-nan col
			idx_prev = tz.loc[:idx-1,col].last_valid_index()
			idx_next = tz.loc[idx+1:,col].first_valid_index()

			if idx_prev is not None and idx_next is not None:

				col_prev = tz.loc[idx_prev,col]
				col_next = tz.loc[idx_next,col]

				# if previous "col" equals next "col" (provided they are from the same rider)
				if col_prev == col_next:
					tz.loc[idx, col] = col_prev
	return tz

def remove_faulty_timezones(tz):
    # probably can automate this:
    # if timezone_dexcom != timezone_trainingpeaks (and both are not nan), 
    # then remove everything up until the previous tz change in trainingpeaks
    # parts where we know the timezone was wrong
    # if it changes later, remove the part for which we don't know until the change
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-02-05') & (tz.date <= '2018-02-18'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-02-21') & (tz.date <= '2018-02-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-02-28') & (tz.date <= '2018-03-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-04-06') & (tz.date <= '2018-04-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-04-21') & (tz.date <= '2018-04-23'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-05-22') & (tz.date <= '2018-05-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-07-10') & (tz.date <= '2018-07-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-08-13') & (tz.date <= '2018-08-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-09-17') & (tz.date <= '2018-09-22'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2018-10-15') & (tz.date <= '2018-10-16'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2019-02-13') & (tz.date <= '2019-02-20'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2019-03-05') & (tz.date <= '2019-03-12'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2019-05-23') & (tz.date <= '2019-05-25'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2019-06-01') & (tz.date <= '2019-06-07'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2019-09-04') & (tz.date <= '2019-09-09'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2019-09-23') & (tz.date <= '2019-09-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2020-01-02') & (tz.date <= '2020-01-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2020-10-20') & (tz.date <= '2020-11-01'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 2) & (tz.date >= '2018-07-31') & (tz.date <= '2018-08-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 2) & (tz.date >= '2018-08-14') & (tz.date <= '2018-08-18'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date >= '2018-12-08') & (tz.date <= '2018-12-09'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date >= '2019-03-06') & (tz.date <= '2019-03-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date >= '2019-03-23') & (tz.date <= '2019-03-25'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date == '2019-05-25'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date >= '2019-09-23') & (tz.date <= '2019-09-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date >= '2019-12-12') & (tz.date <= '2019-12-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 4) & (tz.date >= '2020-01-05') & (tz.date <= '2020-01-12'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 5) & (tz.date >= '2019-01-07') & (tz.date <= '2019-01-10'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date == '2016-12-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2016-12-10') & (tz.date <= '2016-12-11'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2018-05-22') & (tz.date <= '2018-05-29'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2018-06-05') & (tz.date <= '2018-06-11'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2018-08-27') & (tz.date <= '2018-08-28'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2018-11-18') & (tz.date <= '2018-11-22'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2018-11-27') & (tz.date <= '2018-11-28'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2019-03-31') & (tz.date <= '2019-04-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2019-05-26') & (tz.date <= '2019-05-29'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2019-09-29') & (tz.date <= '2019-10-02'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2019-10-06') & (tz.date <= '2019-10-12'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2019-12-12') & (tz.date <= '2019-12-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2020-01-15') & (tz.date <= '2020-01-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2020-02-21') & (tz.date <= '2020-02-22'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 6) & (tz.date >= '2020-03-09') & (tz.date <= '2020-03-11'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2017-03-29') & (tz.date <= '2017-03-30'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2017-06-12') & (tz.date <= '2017-06-15'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2018-10-04') & (tz.date <= '2018-10-10'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2018-10-13') & (tz.date <= '2018-10-14'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date == '2018-10-20'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2018-10-25') & (tz.date <= '2018-10-28'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 10) & (tz.date == '2018-10-30'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 12) & (tz.date == '2018-10-28'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 12) & (tz.date >= '2018-11-26') & (tz.date <= '2018-12-16'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 12) & (tz.date >= '2019-02-17') & (tz.date <= '2019-03-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 12) & (tz.date == '2019-03-31'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 12) & (tz.date == '2019-11-18'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 12) & (tz.date >= '2020-03-02') & (tz.date <= '2020-03-07'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 13) & (tz.date >= '2019-09-28') & (tz.date <= '2019-09-30'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 13) & (tz.date >= '2019-10-16') & (tz.date <= '2019-10-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 13) & (tz.date >= '2020-02-28') & (tz.date <= '2020-03-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date == '2015-11-29'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2015-12-22') & (tz.date <= '2015-12-29'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2016-01-13') & (tz.date <= '2016-02-15'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2016-02-20') & (tz.date <= '2016-02-29'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2016-03-04') & (tz.date <= '2016-03-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2016-05-11') & (tz.date <= '2016-05-24'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2016-05-27') & (tz.date <= '2016-06-01'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date == '2018-09-28'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2018-10-18') & (tz.date <= '2018-11-30'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2018-12-21') & (tz.date <= '2019-01-06'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2019-02-13') & (tz.date <= '2019-02-26'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2019-04-11') & (tz.date <= '2019-07-12'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2019-08-10') & (tz.date <= '2019-08-13'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2019-08-30') & (tz.date <= '2019-10-27'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2019-12-03') & (tz.date <= '2019-12-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2020-02-05') & (tz.date <= '2020-02-10'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 16) & (tz.date >= '2018-11-27') & (tz.date <= '2018-11-30'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2017-08-11') & (tz.date <= '2017-08-13'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2017-08-31') & (tz.date <= '2017-09-02'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2017-09-05') & (tz.date <= '2017-09-07'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2017-09-15') & (tz.date <= '2017-09-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-04-06') & (tz.date <= '2018-04-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-04-20') & (tz.date <= '2018-04-23'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-06-05') & (tz.date <= '2018-07-01'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-07-16') & (tz.date <= '2018-07-22'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-08-03') & (tz.date <= '2018-08-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-08-10') & (tz.date <= '2018-08-12'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-09-19') & (tz.date <= '2018-09-22'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2019-01-31') & (tz.date <= '2019-02-23'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2019-03-05') & (tz.date <= '2019-03-09'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2019-04-08') & (tz.date <= '2019-04-12'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2019-05-23') & (tz.date <= '2019-05-27'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2019-07-29') & (tz.date <= '2019-08-06'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2019-08-26') & (tz.date <= '2019-09-11'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-01-10') & (tz.date <= '2020-01-11'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-03-02') & (tz.date <= '2020-03-08'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-03-12') & (tz.date <= '2020-03-17'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-06-15') & (tz.date <= '2020-06-23'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-06-28') & (tz.date <= '2020-06-30'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-08-29') & (tz.date <= '2020-09-02'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2020-10-31') & (tz.date <= '2020-11-04'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2021-01-24') & (tz.date <= '2021-01-28'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2021-04-26') & (tz.date <= '2021-05-05'), 'timezone'] = np.nan
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2021-06-03') & (tz.date <= '2021-06-08'), 'timezone'] = np.nan
    # TODO: check if all dexcom data from 18 should be shifted a week forward
    # TODO: it seems that maybe dexcom of rider 3 is one date ahead in time?
    return tz

def get_timezones_final(df, root_tp):
	# --------- TrainingPeaks
    # trainingpeaks timezones
    tz_tp = pd.concat({i: pd.read_csv(root_tp+f'/clean/{i}/{i}_timezone_final_list.csv', index_col=0) for i in df.RIDER.unique()})
    tz_tp['date'] = pd.to_datetime(tz_tp['date'])
    tz_tp['timezone'] = pd.to_timedelta(tz_tp['timezone'])
    tz_tp = tz_tp[['date', 'timezone', 'country']]
    # drop impossible timzones
    # TODO: for rider 12, there is still some back and forth italy-spain travelling that cannot be correct
    tz_tp = tz_tp.drop([(3,2030), (3,2031), (4,300), (4,318), (6,82), (6,117), (10, 649), (12,1146), (12, 2053), (12, 2060), (12, 2061),
                        (12, 2240), (12, 2242), (12, 2244), (12, 2246), (12, 2248), (12, 2250), (12, 2253), (12, 2255), (12, 2260), 
                        (12, 2262), (12, 2263), (12, 2265), (12, 2267), (12, 2269), (12, 2273), (12, 2275), (12, 2277), (12, 2280), 
                        (12, 2288), (12, 2290), (12, 2293),
                        (14, 2249), (14, 2252), (14, 2257), (14, 2260)])
    tz_tp = tz_tp.reset_index().rename(columns={'level_0':'RIDER'}).drop('level_1', axis=1).drop_duplicates(subset=['RIDER', 'date'], keep='last')

    # --------- Dexcom
    # dexcom timezones
    tz_dc = df[['RIDER', 'local_timestamp', 'timestamp']]
    tz_dc['date'] = pd.to_datetime(tz_dc['timestamp'].dt.date)
    tz_dc['timezone'] = (tz_dc['local_timestamp'] - tz_dc['timestamp']).round('h')
    tz_dc = tz_dc[['RIDER', 'date', 'timezone']]
    tz_dc = tz_dc.drop_duplicates(subset=['RIDER', 'date'], keep='last').reset_index(drop=True)

    tz_dc = remove_faulty_timezones(tz_dc)

    # --------- Merge
    # fill up trainingpeaks timezones with missing dates
    tz = pd.DataFrame(index=pd.MultiIndex.from_product([df.RIDER.unique(), pd.date_range('2014-01-01', '2021-12-31')], names=['RIDER', 'date'])).reset_index()
    tz = pd.merge(tz, tz_tp, on=['RIDER', 'date'], how='left')
    print("Number of missing timezones: ", tz['timezone'].isna().sum())

    # fillna trainingpeaks timezones with dexcom timezones
    tz = pd.merge(tz, tz_dc, how='left', on=['RIDER', 'date'], suffixes=('_trainingpeaks', '_dexcom'))
    tz = tz.set_index(['date'])
    tz = tz.groupby('RIDER')[['timezone_trainingpeaks', 'timezone_dexcom', 'country']].apply(lambda x: x.loc[x.first_valid_index():x.last_valid_index()]) # remove nans beginning and end
    tz = tz.reset_index()

    print("FILL timezone with trainingpeaks timezone")
    tz['timezone'] = tz['timezone_trainingpeaks']
    print("Number of missing timezones: ", tz['timezone'].isna().sum())
    
    print("FILLNA timezone with dexcom timezone")
    tz['timezone'] = tz['timezone'].fillna(tz['timezone_dexcom'])
    print("Number of missing timezones: ", tz['timezone'].isna().sum())

    print("FILLNA timezone from files before and after")
    tz = tz.groupby('RIDER').apply(lambda x: fill_timezones(x, cols_fill=('timezone', 'country')))
    print("Number of missing timezones: ", tz['timezone'].isna().sum())

    # Correction for countries that were missed
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2019-11-05') & (tz.date <= '2019-11-09'), 'country'] = 'UNK'
    tz.loc[(tz.RIDER == 11) & (tz.date >= '2017-10-26') & (tz.date <= '2017-11-06'), 'country'] = 'China'
    tz.loc[(tz.RIDER == 12) & (tz.date >= '2018-11-18') & (tz.date <= '2018-11-23'), 'country'] = 'UNK'
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2018-08-16') & (tz.date <= '2018-08-26'), 'country'] = 'UNK'

    # change of country
    tz['country_change'] = (tz['country'].shift() != tz['country'])
    # change of Daylight Savings Time
    tz['dst_change'] = False
    for country, times in dst_change.items():
        tz.loc[(tz['country'] == country) & (tz['date'].isin(times)), 'dst_change'] = True

    print("FILLNA timezone from country information")
    # if country hasn't changed and no DST change, fill up timezone
    for idx in tz[tz['timezone'].isna()].index:
        prev_idx = tz.loc[:idx, 'timezone'].last_valid_index()
        next_idx = tz.loc[idx:, 'timezone'].first_valid_index()

        if not tz.loc[idx+1:next_idx].country_change.any() and not tz.loc[idx+1:next_idx].dst_change.any():
            tz.loc[idx, 'timezone'] = tz.loc[next_idx, 'timezone']
        if not tz.loc[prev_idx+1:idx].country_change.any() and not tz.loc[prev_idx+1:idx].dst_change.any():
            tz.loc[idx, 'timezone'] = tz.loc[prev_idx, 'timezone']
    print("Number of missing timezones: ", tz['timezone'].isna().sum())

    # Correct some timezones from TrainingPeaks
    tz.loc[(tz.RIDER == 1) & (tz.date >= '2017-03-11') & (tz.date <= '2017-03-12'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 6) & (tz.date == '2018-10-07'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2017-03-11') & (tz.date <= '2017-03-12'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 12) & (tz.date == '2018-10-27'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2017-03-11') & (tz.date <= '2017-03-12'), 'timezone'] += pd.to_timedelta('1h')

    # Correct some countries from TrainingPeaks
    tz.loc[(tz.RIDER == 3) & (tz.date == '2019-10-22'), 'country'] = 'Spain'
    tz.loc[(tz.RIDER == 4) & (tz.date == '2019-10-20'), 'country'] = 'Japan'
    tz.loc[(tz.RIDER == 5) & (tz.date == '2019-10-22'), 'country'] = 'Netherlands'
    tz.loc[(tz.RIDER == 10) & (tz.date == '2019-03-03'), 'country'] = 'Isle of Man'
    tz.loc[(tz.RIDER == 11) & (tz.date == '2019-02-19'), 'country'] = 'United States'
    tz.loc[(tz.RIDER == 11) & (tz.date == '2019-03-05'), 'country'] = 'United States'
    tz.loc[(tz.RIDER == 12) & (tz.date == '2019-11-06'), 'country'] = 'UNK'
    tz.loc[(tz.RIDER == 13) & (tz.date == '2019-10-24'), 'country'] = 'Hungary'
    tz.loc[(tz.RIDER == 15) & (tz.date == '2018-10-22'), 'country'] = 'France'
    tz.loc[(tz.RIDER == 15) & (tz.date == '2010-10-24'), 'country'] = 'France'
    tz.loc[(tz.RIDER == 17) & (tz.date == '2018-08-09'), 'country'] = 'UNK'
    tz.loc[(tz.RIDER == 18) & (tz.date == '2017-01-23'), 'country'] = 'United States'

    # Check if the timezone has changed when DST change is true
    # In this case, travelling took place at the same time as the DST change
    # However, the travel was within the same timezone, so we can just ffill it.
    dst_error = tz.loc[tz.dst_change & (tz['timezone'].diff() != '-1h') & (tz['timezone'].diff() != '1h')]
    for idx in dst_error.index:
        prev_idx = tz.loc[:idx-1, 'timezone'].last_valid_index()
        tz.loc[prev_idx:idx-1, 'timezone'] = tz.loc[prev_idx:idx-1, 'timezone'].fillna(method='ffill')

    print("FILLNA bfill and ffill timezone until middle between two timezones")
    for j in range(100):
        tz['timezone'] = tz['timezone'].fillna(method='ffill', limit=1)
        tz['timezone'] = tz['timezone'].fillna(method='bfill', limit=1)
    print("Number of missing timezones: ", tz['timezone'].isna().sum())

    # Identify timezone changes due to travelling (i.e. when dst does not change)
    tz['timezone_change'] = (tz['timezone'].shift() != tz['timezone'])

    print("Number of missing countries: ", tz['country'].isna().sum())
    print("FILLNA country from timezone information")
    for idx in tz[tz['country'].isna()].index:
        prev_idx = tz.loc[:idx, 'country'].last_valid_index()
        next_idx = tz.loc[idx:, 'country'].first_valid_index()

        if not tz.loc[idx+1:next_idx].timezone_change.any() and not tz.loc[idx+1:next_idx].dst_change.any():
            tz.loc[idx, 'country'] = tz.loc[next_idx, 'country']
        if not tz.loc[prev_idx+1:idx].timezone_change.any() and not tz.loc[prev_idx+1:idx].dst_change.any():
            tz.loc[idx, 'country'] = tz.loc[prev_idx, 'country']
    # note: countries are automatically forward filled if no timezone change has taken place with this code
    print("Number of missing countries: ", tz['country'].isna().sum())

    print("FILLNA country forward fill")
    tz['country'] = tz['country'].fillna(method='ffill')
    print("Number of missing countries: ", tz['country'].isna().sum())

    # Recalculate dst change (because countries are fillna'd)
    tz['dst_change'] = False
    for country, times in dst_change.items():
        tz.loc[(tz['country'] == country) & (tz['date'].isin(times)), 'dst_change'] = True
    
    # Corrections based on dst_error
    # Note: data of 17 is a mess anyway, so don't do too many corrections
    tz.loc[(tz.RIDER == 2) & (tz.date == '2019-10-26'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 2) & (tz.date == '2020-10-25'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 3) & (tz.date >= '2015-10-23') & (tz.date <= '2015-10-24'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 3) & (tz.date == '2016-10-29'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 3) & (tz.date >= '2021-10-31') & (tz.date <= '2021-11-04'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 5) & (tz.date >= '2021-10-31') & (tz.date <= '2021-11-01'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 7) & (tz.date >= '2016-10-01') & (tz.date <= '2016-10-29'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 7) & (tz.date >= '2018-10-28') & (tz.date <= '2018-11-07'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 9) & (tz.date >= '2014-10-07') & (tz.date <= '2014-10-25'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 9) & (tz.date >= '2018-10-15') & (tz.date <= '2018-10-27'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 9) & (tz.date >= '2020-03-29') & (tz.date <= '2020-04-09'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 10) & (tz.date >= '2021-10-29') & (tz.date <= '2021-10-30'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 12) & (tz.date >= '2014-10-24') & (tz.date <= '2014-10-25'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 13) & (tz.date >= '2019-10-24') & (tz.date <= '2019-10-26'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2014-10-17') & (tz.date <= '2014-10-25'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 14) & (tz.date >= '2020-10-25') & (tz.date <= '2020-10-28'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 15) & (tz.date >= '2016-08-19') & (tz.date <= '2016-10-29'), 'timezone'] += pd.to_timedelta('1h')
    tz.loc[(tz.RIDER == 16) & (tz.date >= '2021-03-21') & (tz.date <= '2021-03-27'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 17) & (tz.date >= '2019-03-16-') & (tz.date <= '2019-03-30'), 'timezone'] += pd.to_timedelta('-1h')
    tz.loc[(tz.RIDER == 17) & (tz.date == '2018-12-04') & (tz.date <= '2019-03-15'), 'country'] = 'France'
    tz.loc[(tz.RIDER == 18) & (tz.date >= '2016-10-24') & (tz.date <= '2016-11-05'), 'timezone'] += pd.to_timedelta('1h')

    dst_error = tz.loc[tz.dst_change & (tz['timezone'].diff() != '-1h') & (tz['timezone'].diff() != '1h')]
    print("Daylight Savings Time error ", dst_error)

    tz = tz.drop(['country_change', 'timezone_change'], axis=1)

    # identify travel: either timezone change or country change, and not dst change
    tz['travel'] = (tz['RIDER'].diff() == 0) & (tz['country'].shift() != tz['country']) | (tz['timezone'].shift() != tz['timezone']) & ~tz['dst_change']
    return tz