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
import gc

import numpy as np
import pandas as pd

from helper import *
from calc import *
from config import rider_mapping

path = 'data/Dexcom/'

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

def insert_rows(df, idx_insert, df_tm, idx_tm):
	"""
	Insert rows of time changes that were not identified with the dexcom timezone changes
	e.g. can happen if an athlete changes transmitters
	
	Arguments:
		df 			dataframe with timezone changes
		df_tm 		dataframe with transmitter changes
		idx_insert	index in (RIDER, n) where to insert in df

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

	return df_A.append(rows).append(df_B)

def insert_transmitter_changes(df_changes, df_transmitter_change):
	# TODO!!!!!
	df_changes = insert_rows(df_changes, (4,9), df_transmitter_change, ((4,2), (4,3)))
	df_changes.loc[(4,9), 'timediff'].iloc[0] = pd.to_timedelta('-8h') # find out manually

	df_changes = insert_rows(df_changes, (6,55), df_transmitter_change, ((6,2), (6,3)))
	df_changes.loc[(6,55), 'timediff'].iloc[0] = pd.to_timedelta('-8h') # find out manually

	df_changes = insert_rows(df_changes, (15,0), df_transmitter_change, ((15,1), (15,2)))
	df_changes.loc[(15,0), 'timediff'].iloc[0] = pd.to_timedelta('1h') # find out manually

	# Reset counter n
	df_changes.reset_index(inplace=True)
	df_changes['n'] = df_changes.groupby('RIDER').cumcount()
	df_changes.set_index(['RIDER', 'n'], inplace=True)

def get_timezone_changes():

	if not os.path.exists(path+'timezone/'):
		os.mkdir(path+'timezone/')

	# read data
	df = pd.read_csv(path+'dexcom_sorted.csv', index_col=0)
	df.local_timestamp = pd.to_datetime(df.local_timestamp)

	# calculate timezones only for EGV readings
	df_egv = df[df['Event Type'] == 'EGV']
	df_egv = df_egv.reset_index()

	# identify timezone windows for each rider
	df_egv = calc_timezone_diff(df_egv)
	df_egv = calc_timezone_change(df_egv)

	# create list of timezone changes
	df_changes = get_timezone_changelist(df_egv)

	# add missing timezone changes
	# sometimes we don't observe a timezone change when an athlete was not wearing a transmitter 
	# and was in between transmitters
	df_transmitter_change = get_transmitter_changelist(df_egv)
	df_changes = insert_transmitter_changes(df_changes, df_transmitter_change)

	# TODO!!!!!!!!!!!!!!!!!!!!!!!!!!
	# starting timezones
	# these are chosen in such a way that the final timezones overlap (roughly) 
	# with the actual timezone changes observed in trainingpeaks
	df_changes.loc[(1,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(2,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(3,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(4,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(5,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(6,0), 'timezone'] = pd.to_timedelta('1 day 01:00:00')
	df_changes.loc[(10,0), 'timezone'] = pd.to_timedelta('0h')
	df_changes.loc[(12,0), 'timezone'] = pd.to_timedelta('8h')
	df_changes.loc[(13,0), 'timezone'] = pd.to_timedelta('1h')
	df_changes.loc[(14,0), 'timezone'] = pd.to_timedelta('3h')
	df_changes.loc[(15,0), 'timezone'] = pd.to_timedelta('1h')

	# -------- Calculate final timezones
	# calculate final timezone
	for i in df_changes.index.get_level_values(0).unique():
		for n in df_changes.loc[i].index[1:]:
			df_changes.loc[(i,n), 'timezone'] = df_changes.loc[(i,n-1), 'timezone'] + df_changes.loc[(i,n-1), 'timediff']

	df_changes.loc[:,'index_max'].iloc[-1] = df.index[-1]
	df_changes.loc[:,'local_timestamp_max'].iloc[-1] = df.iloc[-1].local_timestamp

	df_changes.index_max = df_changes.index_max.astype(int)

	df_changes.drop('timediff', axis=1, inplace=True)
	df_changes.to_csv(path+'timezone/timezone_dexcom.csv')
	return df_changes

if __name__ == '__main__':
    get_timezone_changes()