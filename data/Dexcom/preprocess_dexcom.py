# TODO: check if we indeed can use event subtype to fill extremes with
# TODO: watch out for filtering by Event Type!!
# TODO: preprocess glucose values (look online on internet)
# TODO: also look for sensor errors
import os
import gc
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from pandas_profiling import ProfileReport

from helper import *
from calc import *
from config import rider_mapping

path = './'
path_tz = '../TrainingPeaks/2019/timezone/'

if not os.path.exists(path+'drop/'):
	os.mkdir(path+'drop/')

df = pd.read_csv('TNN_CGM_2019.csv', index_col=0)

df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

# sort and save old index as backup
df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(path+'dexcom_raw.csv')

# --------------------- clean glucose
# remove text "high" and "low" from glucose values
df['Glucose Value (mmol/L) EXTREME'] = df['Glucose Value (mmol/L)'].apply(lambda x: x if isinstance(x, str) else np.nan)
df['Glucose Value (mg/dL) EXTREME'] = df['Glucose Value (mg/dL)'].apply(lambda x: x if isinstance(x, str) else np.nan)

df['Glucose Value (mmol/L)'] = pd.to_numeric(df['Glucose Value (mmol/L)'], errors='coerce')
df['Glucose Value (mg/dL)'] = pd.to_numeric(df['Glucose Value (mg/dL)'], errors='coerce')

# fill up glucose values mg/dL with mmol/L
df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].fillna(df['Glucose Value (mmol/L)'] * mmoll_mgdl)
df['Glucose Value (mg/dL) EXTREME'] = df['Glucose Value (mg/dL) EXTREME'].fillna(df['Glucose Value (mmol/L) EXTREME']) 
print("FILLNA Glucose Value mg/dL with mmol/L to mg/dL")
print("FILLNA Glucose Value mg/dL EXTREME with mmol/L")

df.drop(['Glucose Value (mmol/L)', 'Glucose Value (mmol/L) EXTREME'], axis=1, inplace=True)

# NOTE: there will be a lot of duplicates in the file now, as US saved the data in mg/dL and EU saved it sometimes in mmol/L
# In the merge, these were considered separate rows, even though they should be merged.

# clean zeros glucose (sometimes a zero is put in instead of nan, when there is no glucose measurement)
df.loc[(df['Event Type'] != 'EGV') & (df['Event Type'] != 'Calibration'), 'Glucose Value (mg/dL)'] = np.nan
print("CHECK Are there remaining zero glucose values: ", not df[df['Glucose Value (mg/dL)'] == 0].empty)

# Note: sometimes event subtype is "Low" or "High"
# It seems that in this is when it is too low or too high to measure
mask_subtype = (df['Event Type'] == 'EGV') & df['Event Subtype'].notna()
print("CHECK Event Subtypes for Event Type == EGV: ", 
	df[mask_subtype]['Event Subtype'].unique())
print("CHECK Number of subtype entries for Event Type == EGV: ", 
	mask_subtype.sum())
print("CHECK Number of subtype entries for Event Type == EGV where Glucose Value (mg/dL) EXTRME is nan: ",
	(mask_subtype & df['Glucose Value (mg/dL) EXTREME'].isna()).sum())
print("CHECK Event subtype anywhere unequal to EXTREME: ", 
	not df[mask_subtype & df['Glucose Value (mg/dL) EXTREME'].notna() 
	& (df['Event Subtype'] != df['Glucose Value (mg/dL) EXTREME'])].empty)
df.loc[mask_subtype, 'Glucose Value (mg/dL) EXTREME'] = df.loc[mask_subtype, 'Glucose Value (mg/dL) EXTREME']\
															.fillna(df.loc[mask_subtype, 'Event Subtype']) 
df.loc[mask_subtype, 'Event Subtype'] = np.nan
print("FILLNA Glucose Value mg/dL EXTREME with Event Subtype")

# replace low and high with 40 and 400
df['Glucose Value (mg/dL)'].fillna(df['Glucose Value (mg/dL) EXTREME'].replace({'Low':40., 'High':400.}), inplace=True)
df.drop('Glucose Value (mg/dL) EXTREME', axis=1, inplace=True)
df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].astype(float)
print("REPLACE Low with 40 and High with 400")
print("FILLNA Glucose Value mg/dL with EXTREME 40 (Low) and 400 (High)")

# --------------------- manual timezone mistakes riders
# correct for mistakes by riders in manually switching timezones of their recorder
# 15: mistake here is that he had to reset it after travelling
# and then he switched the month and day around (e.g. 10/7 instead of 7/10)
df.loc[(df.RIDER == 15)\
	& (df['Source Device ID'] == 'PL82501061')\
	& (df['Transmitter ID'] == '80YBT4')\
	& (df['Transmitter Time (Long Integer)'] >= 4163883)\
	& (df['Transmitter Time (Long Integer)'] <= 6179527), 'local_timestamp'] += pd.to_timedelta('90days')
print("FIX timestamps of 15 from 7/10 to 10/7")

# 4: changed it to the wrong month
# this is calculated by subtracting the transmitter timediff
df.loc[(df.RIDER == 4)\
	& (df['Source Device ID'] == 'PL82609380')\
	& (df['Transmitter ID'] == '810APT')\
	& (df['Transmitter Time (Long Integer)'] >= 6233271)\
	& (df['Transmitter Time (Long Integer)'] <= 6879458), 'local_timestamp'] += pd.to_timedelta('30days 23:55:05') 
print("FIX timestamps of 4 october -> november")

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)

# TODO: check if it also holds for the calibration measurements in the same time window

# 12: strong suspicions that he has his device setup with the wrong month in there
# because for both 2018 and 2019 the japan cup the timezone change took place exactly one month later


# --------------------- error transmitter ID
# OBSERVATION We see the following:
# - There is a time window when both the first and second transmitter are observed, alternating a bit
# - In the time that we observed both the first and second transmitter, 
#   the transmitter time of the first transmitter continues. 
# - The transmitter time of the second transmitter is at some point reset to zero (or 7500)

# CONCLUSION Therefore we conclude that the riders probably continued using the first transmitter
# (longer than they should have?) and this messed up the system.

# SOLUTION The solution is to change the transmitter ID in the period that we observe both the old
# and the new transmitter, to only the ID of the old transmitter. Then all issues should be fixed.

# '2019-03-05 10:50:39' '2019-03-27 05:24:16' '80LF01' '80QJ2F'
# see 6_transmitter_80QJ2F_time_reset.png
df.loc[(df.RIDER == 6) & (df.local_timestamp <= '2019-03-27 05:24:16')\
	& (df['Transmitter ID'] == '80QJ2F') & (df['Event Type'] == 'EGV'), 'Transmitter ID'] = '80LF01'
print("FIX (6) transmitter ID between 2019-03-05 10:50:39 and 2019-03-27 05:24:16 from 80QJ2F to 80LF01")

# '2019-08-22 21:25:41' '2019-09-24 20:59:16' '80UKML' '80RE8H'
# see 6_transmitter_80RE8H_time_reset.png
df.loc[(df.RIDER == 6) & (df.local_timestamp <= '2019-09-24 20:59:16')\
	& (df['Transmitter ID'] == '80RE8H') & (df['Event Type'] == 'EGV'), 'Transmitter ID'] = '80UKML'
print("FIX (6) transmitter ID between 2019-08-22 21:25:41 and 2019-09-24 20:59:16 from 80RE8H to 80UKML")

# '2019-01-17 15:56:19' '2019-01-18 22:36:15' '80JPC8' '80RRBL'
# see 14_transmitter_80RRBL_time_reset.png
df.loc[(df.RIDER == 14) & (df.local_timestamp <= '2019-01-18 22:36:15')\
	& (df['Transmitter ID'] == '80RRBL') & (df['Event Type'] == 'EGV'), 'Transmitter ID'] = '80JPC8'
print("FIX (14) transmitter ID between 2019-01-17 15:56:19 and 2019-01-18 22:36:15 from 80RRBL to 80JPC8")

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)


# OBSERVATION Nan transmitter ID after taking a break
# It takes a bit for the device to register the transmitter ID
# You can see this by the fact that the transmitter time continues counting correctly after the nan transmitter IDs

df_nan_transmitter = df[df['Transmitter ID'].isna() & (df['Event Type'] == 'EGV')]
idx_first_nan_transmitter = df_nan_transmitter[df_nan_transmitter.index.to_series().diff() != 1].index
idx_last_nan_transmitter = df_nan_transmitter[df_nan_transmitter.index.to_series().diff().shift(-1) != 1].index

# transmitter ID not saved yet with new transmitter
# note that there are no calibration measurements in this slice
# otherwise we would have to filter for it
for i in range(len(idx_first_nan_transmitter)):	
	df.loc[idx_first_nan_transmitter[i]:idx_last_nan_transmitter[i], 'Transmitter ID'] = df.loc[idx_last_nan_transmitter[i]+1, 'Transmitter ID']
	print("FIX nan transmitter ID between index %s and %s"%(idx_first_nan_transmitter[i], idx_last_nan_transmitter[i]))

# Note that for the last transmitter, there is the same observation of above (the jump in time)
# note: no calibration measurement in slice (luckily)
idx_last_break = df[df['Transmitter Time (Long Integer)'].diff() < 0].loc[idx_first_nan_transmitter[1]:idx_last_nan_transmitter[1]].index[0] - 1
df.loc[idx_first_nan_transmitter[1]:idx_last_break, 'Transmitter ID'] = 'UNK_ID'
print("FIX (15) transmitter ID between 2019-07-21 14:27:36 and 2019-08-20 13:51:12 from 80YBT4 to UNK_ID")

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)

"""
# CHECK
matplotlib.use('TkAgg')

for i in df['RIDER'].unique():
	df_i = df[df.RIDER == i]
	sns.lineplot(data=df_i, x='local_timestamp', y='Transmitter Time (Long Integer)', hue='Transmitter ID')
	plt.show()

for _, (i, tid) in df[['RIDER', 'Transmitter ID']].drop_duplicates().iterrows():
	print(i, tid)
	df_t = df[(df.RIDER == i) & (df['Transmitter ID'] == tid)]
	plt.plot(df_t['local_timestamp'], df_t['Transmitter Time (Long Integer)'])
	plt.show()
	plt.close()
"""
# 2 81J1XS (stayed constant for a bit)
# 4 80CPYD (stayed constant for a bit, then went up much faster and then constant for a bit again)
# 4 810APT (some back and forth for a day it seems (probably device change))
# 4 8HLEH9 (stayed constant for a bit)
# 6 80CPX2 (sth weird going on here)
# 12 809SED (some back and forth for some days it seems (probably device change))
# 15 80YBT4 (some back and forth for some days it seems (probably device change))

# --------------------- date range
# filter date range
df = df[(df.local_timestamp.dt.date < datetime.date(2019,12,1)) & (df.local_timestamp.dt.date >= datetime.date(2018,12,1))]
print("DROPPED entries after 30-11-2019 or before 01-12-2018")

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)', 'source'], inplace=True)
df.reset_index(drop=True, inplace=True)

# --------------------- duplicates
# drop duplicates rows
df_dupl = df[df.drop('source', axis=1).duplicated(keep=False)]
df_dupl.to_csv(path+'drop/duplicated_rows.csv')
print("DROP %s duplicated rows"%df.drop('source', axis=1).duplicated().sum())
df = df[~df.drop('source', axis=1).duplicated(keep='first')] ; del df_dupl ; gc.collect()
# TODO: delete dupl folder and all associated files

# drop duplicates rows where glucose value is not exactly the same, but the rest is
# (this includes transmitter time, so we do not drop duplicates from travelling)
df_dupl = df[df.drop(['source', 'Glucose Value (mg/dL)'], axis=1).duplicated(keep=False)]
df = df[~df.drop(['source', 'Glucose Value (mg/dL)'], axis=1).duplicated(keep=False)]

# rows with originally glucose values in different units, therefore not merged in other script
print(df_dupl.notna().sum()) # print columns that we don't need to use
df_gb = df_dupl.groupby(['RIDER', 'Event Type', 'local_timestamp', 'Transmitter Time (Long Integer)', 
	'Transmitter ID', 'Source Device ID',]).agg({'Glucose Value (mg/dL)':'mean', 'source':'first'}).reset_index()
df = df.append(df_gb, ignore_index=True)

# remaining readings without trainsmitter id or transmitter time (e.g. calibration readings)
df_res = df_dupl[~df_dupl.local_timestamp.isin(df_gb.local_timestamp)]
df_gb = df_res.groupby(['RIDER', 'Event Type', 'local_timestamp', 'Source Device ID',]).agg({'Glucose Value (mg/dL)':'mean', 'source':'first'}).reset_index()
df = df.append(df_gb, ignore_index=True)

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)

# check if there are duplicated EGV readings with the same rider, transmitter_order, transmitter_time
print("CHECK Number of duplicated EGV readings (rider, transmitter id, transmitter time): ", 
	df[df['Event Type'] == 'EGV'].duplicated(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)'], keep=False).sum())
df_dupl = df[(df['Event Type'] == 'EGV') & df.duplicated(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)'], keep=False)]
df.drop(df_dupl.index, inplace=True)

df_dupl.sort_values(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)', 'source'], inplace=True)
df_gb = df_dupl.groupby(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)'])\
	.agg({'local_timestamp':'first',
		  'Event Type':'first', 
		  'Source Device ID':'first',
		  'Glucose Value (mg/dL)':'first', # note: we tested and the max difference between the two was 1mg/dl
		  'source':'first'}).reset_index()
# Note: we also just could have selected the iphone/EU data here
df = df.append(df_gb, ignore_index=True)

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)

del df_dupl, df_gb, df_res

# --------------------- nans
# check for nan rows
df_nan = df[(df[['Insulin Value (u)', 'Carb Value (grams)', 'Duration (hh:mm:ss)', 'Glucose Value (mg/dL)']].isna().all(axis=1)
			& (df['Event Type'] != 'Insulin') & (df['Event Type'] != 'Health'))]
df_nan.to_csv(path+'drop/nan_rows.csv')
print("DROP %s nan rows (not event type insulin or health)"%len(df_nan))
df.drop(df_nan.index, inplace=True) ; del df_nan ; gc.collect()

# --------------------- transmitter check
# check if there are any readings that are not EGV or Calibration and that do have a transmitter ID
print("Number of readings that are not EGV or Calibration and that do have a transmitter ID: ", 
	((df['Event Type'] != 'EGV') & (df['Event Type'] != 'Calibration') & df['Transmitter ID'].notna()).sum())

# find out order of the transmitters, and then sort them
# such that the actual first transmitted signal will always be higher (regardless of local timestamps)
# this can later be used to identify if there is any unidentified travelling still in the data
df_transmitter = df[df['Event Type'] == 'EGV'].groupby(['RIDER', 'Transmitter ID'])\
	.agg({'local_timestamp':['min', 'max']})\
	.sort_values(['RIDER', ('local_timestamp', 'min')])

# Check if there is overlapp between transmitters (of the same rider)
# Note: there is only overlap if you include Calibrations
# because the riders double up when the transmitter is at the end of its lifetime.
# They also use often the next transmitter as a calibrator.
for i in df_transmitter.index.get_level_values(0).unique():
	for j, tid in enumerate(df_transmitter.loc[i].index):
		try:
			date_curr_max = df_transmitter.loc[i].loc[tid, ('local_timestamp', 'max')]
			date_next_min = df_transmitter.loc[i].iloc[j+1][('local_timestamp', 'min')]

			# if there is overlap
			if date_curr_max >= date_next_min:
				print("Overlap transmitters")
				print(i, df_transmitter.loc[i].iloc[j:j+2])

		except IndexError:
			pass

# Create transmitter order
transmitter_order = {df_transmitter.index.get_level_values(1)[n]:n for n in np.arange(len(df_transmitter))}
df['transmitter_order'] = df['Transmitter ID'].apply(lambda x: transmitter_order[x] if x in transmitter_order.keys() else len(transmitter_order))
del transmitter_order ; gc.collect()

# Split in EGV and non-EGV for sorting
df_egv = df[df['Event Type'] == 'EGV']
df_nonegv = df[df['Event Type'] != 'EGV']

# Sort by: Event Type - RIDER - transmitter_order - Transmitter Time
df_egv.sort_values(by=['RIDER', 'transmitter_order', 'Transmitter Time (Long Integer)', 'Source Device ID'], inplace=True)
df = df_egv.append(df_nonegv)
df.reset_index(drop=True, inplace=True)

# For each non-EGV reading, put it in the right rider + time window
for idx, (i, t) in df.loc[df['Event Type'] != 'EGV', ['RIDER', 'local_timestamp']].iterrows():
	loc = df.index.get_loc(idx)

	# TODO: what if during travelling?
	idx_new = df[(df.RIDER == i) & (df.local_timestamp < t) & (df['Event Type'] == 'EGV')].index[-1]
	loc_new = df.index.get_loc(idx_new)

	df = df.loc[np.insert(np.delete(df.index, loc), loc_new+1, loc)]
df.reset_index(drop=True, inplace=True)

df.to_csv(path+'dexcom_sorted.csv')

# --------------------- RUN preprocess_timezone_dexcom.py here

# --------------------- Transform to local time
df = pd.read_csv(path+'dexcom_sorted.csv', index_col=0)
df.local_timestamp = pd.to_datetime(df.local_timestamp)

# read timezones extracted from the trainingpeaks data
df_changes = pd.read_csv('timezone/timezone_dexcom.csv', index_col=[0,1])
df_changes.timezone = pd.to_timedelta(df_changes.timezone)
df_changes.local_timestamp_min = pd.to_datetime(df_changes.local_timestamp_min)
df_changes.local_timestamp_max = pd.to_datetime(df_changes.local_timestamp_max)

for (i,n), (idx_min, idx_max, _, _, tz) in df_changes.iterrows():
	df.loc[idx_min:idx_max, 'timestamp'] = df.loc[idx_min:idx_max, 'local_timestamp'] - tz

df.sort_values(['RIDER', 'timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)

# check if it worked
df.loc[df['Event Type'] == 'EGV', 'timestamp_diff'] = df.loc[df['Event Type'] == 'EGV', 'timestamp'].diff()
df.loc[df['Event Type'] == 'EGV', 'transmitter_diff'] = df.loc[df['Event Type'] == 'EGV', 'Transmitter Time (Long Integer)'].diff()

df['timediff'] = df['timestamp_diff'] - pd.to_timedelta(df['transmitter_diff'], 'sec')
df.loc[df['transmitter_order'].diff() != 0, 'timediff'] = np.nan # correct for transmitter change

df['gap'] = (df['timediff'] > '5min')
print("Number of gaps left: ", df['gap'].sum())

df['dup'] = (df['timediff'] < '-5min')
print("Number of dups left: ", df['dup'].sum())

df['change'] = df['dup'] | df['gap']
print("Number of changes left: ", df['change'].sum())

print("When transmitter time goes down: ",
	df[(df['transmitter_diff'] < 0) & (df.RIDER.diff() == 0) & (df['transmitter_order'].diff() == 0) & (df['Event Type'].shift() == df['Event Type']) & (df['Event Type'] == 'EGV')])
# TODO: it seems that there is only one weird measurement: 322039 and 322471
# I think these should be calibration measurements, but they are marked as EGV
# 322471	6	2018-12-18 18:25:02
# 322039	6	2018-12-17 06:35:09

"""
matplotlib.use('TkAgg')

for i in df['RIDER'].unique():
	df_i = df[df.RIDER == i]
	sns.lineplot(data=df_i, x='timestamp', y='Transmitter Time (Long Integer)', hue='Transmitter ID')
	plt.show()

for _, (i, tid) in df[['RIDER', 'Transmitter ID']].drop_duplicates().iterrows():
	df_t = df[(df.RIDER == i) & (df['Transmitter ID'] == tid)]
	#if not df_t['Transmitter Time (Long Integer)'].is_monotonic:
	if df_t['change'].any():
		df_t_err = df_t[df_t['change']]
		df_t.to_csv('timezone/check_%s_%s.csv'%(i,tid))
		df_t_err.to_csv('timezone/check_%s_%s_err.csv'%(i,tid))
		print(i, tid, " ERROR")
		plt.plot(df_t['timestamp'], df_t['Transmitter Time (Long Integer)'])
		plt.scatter(df_t_err['timestamp'], df_t_err['Transmitter Time (Long Integer)'])
		plt.show()
		plt.close()
"""

df.drop(['transmitter_order', 'transmitter_diff',  'timestamp_diff', 'timediff', \
	'gap', 'dup', 'change'], axis=1, inplace=True)
df.to_csv(path+'dexcom_utc.csv')

# --------------------- RUN preprocess_timezone_trainingpeaks_dexcom.py here

# --------------------------------- UTC to local
df_changes = pd.read_csv('../TrainingPeaks+Dexcom/timezone/timezone_changes_final.csv', index_col=[0,1])

df_changes.timezone = pd.to_timedelta(df_changes.timezone)
df_changes.timestamp_min = pd.to_datetime(df_changes.timestamp_min)
df_changes.timestamp_max = pd.to_datetime(df_changes.timestamp_max)
df_changes.local_timestamp_min = pd.to_datetime(df_changes.local_timestamp_min)
df_changes.local_timestamp_max = pd.to_datetime(df_changes.local_timestamp_max)

df = pd.read_csv(path+'dexcom_utc.csv', index_col=0)
df.local_timestamp = pd.to_datetime(df.local_timestamp)
df.timestamp = pd.to_datetime(df.timestamp)

# recalculate local timestamp
df.rename(columns={'local_timestamp':'local_timestamp_raw'}, inplace=True)

for (i,n), (tz, _, _, ts_min, ts_max, _, _) in df_changes.iterrows():
	mask_tz = (df.RIDER == i) & (df.timestamp >= ts_min) & (df.timestamp <= ts_max)
	df.loc[mask_tz, 'local_timestamp'] = df.loc[mask_tz, 'timestamp'] + tz

# TODO: fix all insulin and carbs metrics

df.to_csv(path+'dexcom_clean.csv')