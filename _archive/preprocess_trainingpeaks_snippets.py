# -------------------- Zeros power meter
# TODO: figure out what to do with zeros from power meter
# CHECK: is the previous value for a nan always a zero: answer NO
# CHECK: is the a zero always followed by a nan: answer more NO
# Note: sometimes the power is nan if there is a shift in timestamps
# Note: zero power often happens for negative grades, so maybe it is actually when they stop pedalling
# TODO: ask how this works in real life
# TODO: nan seems to happen a lot also on a specific day, remove that file

# TODO: filter out training sessions for which a whole column is missing
# TODO: remove training sessions for which there is no distance at all

# TODO: find out what happened when there are large gaps in the data
# TODO: calculate values (statistics) on a training-level
# TODO: elevation_gain, acceleartion and accumulated power now contain extreme values when a large shift in time is made within a training
# imputation
"""
for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(path+'clean2/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	# -------------------- Position
	# check if position long and lat are missing at the same time or not 
	print("Number of missing values position_lat: ", df['position_lat'].isna().sum())
	print("Number of missing values position_long: ", df['position_long'].isna().sum())
	print("Number of times both position_lat and position_long missing: ", 
		(df['position_lat'].isna() & df['position_long'].isna()).sum())
	# not really relevant because we're not going to do anything with it anymore

	
	print("\n-------- Remove first rows mostly nan")
	# TODO: move this to the next stage
	# for each file, go over the first rows and check the percentage of nans
	df['percentage_nan'] = df[set(df.columns) - cols_ignore].isna().sum(axis=1) / len(set(df.columns) - cols_ignore)
	count_drop_firstrows = 0
	for f in df.file_id.unique():
		for j, row in df[df.file_id == f].iterrows():
			if row['percentage_nan'] > .75:
				df.drop(j, inplace=True)
				count_drop_firstrows += 1
			else:
				break
	print("DROPPED: {:g} first rows with more than 75\% nans".format(count_drop_firstrows))
	

	# -------------------- Impute nans
	# TODO fill up nans with duplicates
	# df_dupl = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_data_dupl.csv', index_col=0)


	# -------------------- Temperature - TODO maybe move this to before
	# smooth temperature (because of binned values) (with centering implemented manually)
	# rolling mean of {temp_window} seconds, centered
	temp_window = 200 #in seconds
	df_temperature_smooth = df.set_index('local_timestamp')['temperature']\
		.rolling('%ss'%temp_window, min_periods=1).mean()\
		.shift(-temp_window/2, freq='s').rename('temperature_smooth')
	df = pd.merge(df, df_temperature_smooth, left_on='local_timestamp', right_index=True, how='left')
	del df_temperature_smooth ; gc.collect()
	print("CREATED: temperature smooth")

	print("Fraction of rows for which difference between original temperature and smoothed temperature is larger than 0.5: ",
		((df['temperature_smooth'] - df['temperature']).abs() > .5).sum() / df.shape[0])

	PlotPreprocess('../Descriptives/', athlete=i).plot_smoothing(df, 'temperature', kwargs=dict(alpha=.5, linewidth=2))

	# note that 200s should probably be more if you look at the distributions
	sns.histplot(df, x='temperature', kde=True)
	sns.histplot(df, x='temperature_smooth', kde=True, color='red')
	plt.show()
	plt.close()

	# -------------------- Battery level
	# find out if/when battery level is not monotonically decreasing
	# TODO: why does this happen?
	battery_monotonic = pd.Series()
	for idx in df.file_id.unique():
		battery_monotonic.loc[idx] = (df.loc[df.file_id == idx, 'battery_soc'].dropna().diff().fillna(0) <= 0).all()

	if (~battery_monotonic).sum() > 0:
		print("WARNING: Number of trainings for which battery level is not monotonically decreasing: ",
			(~battery_monotonic).sum())
		battery_notmonotonic_index = df.file_id.unique()[~battery_monotonic]
	
		# plot battery levels
		cmap = matplotlib.cm.get_cmap('viridis', len(battery_notmonotonic_index))
		ax = plt.subplot()
		for c, idx in enumerate(battery_notmonotonic_index): #for date in df.date.unique():
			df[df.file_id == idx].plot(ax=ax, x='time_training', y='battery_soc',
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
		plt.ylabel('battery_soc')
		plt.show()
		plt.close()

	# linearly interpolate battery level (with only backwards direction)
	for idx in df.file_id.unique():
		df.loc[df.file_id == idx, 'battery_soc_ilin'] = \
		df.loc[df.file_id == idx, 'battery_soc']\
		.interpolate(method='time', limit_direction='forward')
	print("CREATED: linearly interpolated battery level")

	PlotPreprocess('../Descriptives/', athlete=i).plot_interp(df, 'battery_soc', kwargs=dict(alpha=.5), 
		ikwargs=dict(alpha=.5, kind='scatter', s=10.))

	# TODO: clean more features here, and add some as well

	df.to_csv(path+'clean3/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)
"""


# when timezone_loc == nan and timezone != nan
#df_tz.loc[df_tz['timezone_loc'].isna() & df_tz['timezone'].notna(), 'timezone_combine'] = df_tz['timezone']

# when timezone == nan and timezone_loc != nan
#df_tz.loc[df_tz['timezone'].isna() & df_tz['timezone_loc'].notna(), 'timezone_combine'] = df_tz['timezone_loc']

# when timezone == nan and timezone_loc == nan
df_tz['timezone_change'] = (df_tz['timezone'].diff() != '0d') & df_tz['timezone'].diff().notna()
df_tz['timezone_change_loc'] = (df_tz['timezone_loc'].diff() != '0d') & df_tz['timezone_loc'].diff().notna()

df_tz.loc[df_tz['timezone_change'] & df_tz.timezone_combine.isna(), 'timezone_combine'] = df_tz['timezone']
df_tz.loc[df_tz['timezone_change_loc'] & df_tz.timezone_combine.isna(), 'timezone_combine'] = df_tz['timezone_loc']

# fix zwift files (assume they don't take zwift for travelling so only use at home)
# note: they are all only wintertime, so just lazy solution here
# note: also ZWIFT measurements that are not nan are used for everyone at least once in wintertime
print("NAN tz before: ", df_tz.timezone_combine.isna().sum())
nan_zwift = df_tz.loc[df_tz.timezone_combine.isna() & (df_tz.device_0 == 'ZWIFT'), ['RIDER', 'file_id', 'date']]
tz_zwift = df_tz.loc[df_tz.device_0 == 'ZWIFT', ['RIDER', 'timezone_combine']].dropna()\
	.groupby('RIDER').agg({'timezone_combine':'min'})
for _, (i, f, d) in nan_zwift.iterrows():
	mask = (df_tz.RIDER == i) & (df_tz.file_id == f)
	df_tz.loc[mask, 'timezone_combine'] = tz_zwift.loc[i][0]
print("NAN tz after: ", df_tz.timezone_combine.isna().sum())

# TODO: note is not always going to work, might differ 1 hour roughly.
df_tz['timezone_combine'] = df_tz['timezone_combine'].fillna(method='ffill')

df_tz['timezone_change_combine'] = df_tz['timezone_combine'].diff() != '0d'
df_tz['timezone_diff_combine'] = df_tz['timezone_combine'].diff()



	# fix zwift timestamps
	print("\n-------- Timestamp: impute zwift nan")
	if 'device_ZWIFT' in df:
		ts_zwift = df.loc[df['device_ZWIFT'] & df['local_timestamp'].isna() & df['local_timestamp_loc'].isna(), 'timestamp'].dt.date.unique()
		print("Get timezone of Zwift files for the following dates: ")
		for ts in ts_zwift:
			print(ts)
			ts_range = [ts-datetime.timedelta(days=2), ts-datetime.timedelta(days=1), ts, ts+datetime.timedelta(days=1), ts+datetime.timedelta(days=2)]
			df_ts_zwift = df[df['timestamp'].dt.date.isin(ts_range)]
			tz_zwift = pd.concat([(df_ts_zwift['local_timestamp'] - df_ts_zwift['timestamp']),
								  (df_ts_zwift['local_timestamp_loc'] - df_ts_zwift['timestamp'])], axis=1)
			tz_meta = pd.DataFrame(tz_zwift[0].unique()).dropna().to_numpy()[0]
			tz_loc = pd.DataFrame(tz_zwift[1].unique()).dropna().to_numpy()[0]
			# make sure no difference in ts_range and we have tz both from meta and loc
			if len(tz_meta) == 1 and len(tz_loc) == 1:
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'local_timestamp_loc'] = \
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'timestamp'] + tz_loc[0]
				print("local_timestamp_loc changed to UTC offset: ", tz_loc[0].astype('timedelta64[h]'))
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'local_timestamp'] = \
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'timestamp'] + tz_meta[0]
				print("local_timestamp changed to UTC offset: ", tz_meta[0].astype('timedelta64[h]'))
			else:
				print("local_timestamp_loc not changed because of difference in timezones in the range of two days before until two days after")

	# distribution of timezones
	plt.hist([(df.local_timestamp_loc - df.timestamp).astype('timedelta64[h]').to_numpy(),
			  (df.local_timestamp - df.timestamp).astype('timedelta64[h]').to_numpy()],
			  label=['position', 'metadata'])
	plt.xlabel('UTC offset (hours)')
	plt.legend()
	plt.savefig('../../../Descriptives/hist_timezone_%s.pdf'%i, bbox_inches='tight')
	plt.savefig('../../../Descriptives/hist_timezone_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.close()

	#df['drop_nan_timestamp'] = df['local_timestamp'].isna() & df['local_timestamp_loc'].isna()
	df_nan_ts = df[df['local_timestamp'].isna() & df['local_timestamp_loc'].isna()]
	df.drop(df_nan_ts.index, inplace=True) 
	print("DROPPED: %s files with %s nan local timestamps (both meta and loc)"%(len(df_nan_ts.file_id.unique()), len(df_nan_ts)))

	print("\n-------- Timestamp local combine")
	# combine both local timestamps
	# keep timestamp from location as the primary timestamp
	df['local_timestamp_loc'].fillna(df['local_timestamp'], inplace=True)
	df.drop('local_timestamp', axis=1, inplace=True)
	df.rename(columns={'local_timestamp_loc':'local_timestamp'}, inplace=True)

	# TODO: look into dropping duplicate timestamps for which the local timestamp is incorrect
	# maybe sort by error timestamps as well?
	#print("Error timestamps in duplicate timestamps: ",
	#	df[dupl_timestamp_both & df['keep_devices']].error_local_timestamp.sum() != 0)
	#print("DROPPED: duplicate timestamps for which the local timestamp is possibly incorrect")
	# TODO: for each file, check whether it's duplicate has more nans (so for missing files, check percentage of nans)


#####################################################################################################
	idx_zero_power = df[df.power == 0].index
	idx_nan_power = df[df.power.isna()].index

	idx_first_zero_power = idx_zero_power[idx_zero_power.to_series().diff() != 1]
	idx_first_nan_power = idx_nan_power[idx_nan_power.to_series().diff() != 1]

	df.loc[idx_first_nan_power-1, 'power']
	df['timediff'] = df.local_timestamp.diff() 
	df['power_shift'] = df.power.shift(1)
	df.loc[idx_first_nan_power, ['timediff', 'device_ELEMNT', 'device_ELEMNTBOLT', 'device_GARMIN', 'device_ZWIFT']]
	df.groupby(['device_ELEMNT', 'device_ELEMNTBOLT', 'device_GARMIN', 'device_ZWIFT'])['timediff'].mean()

	df.loc[idx_first_nan_power].loc[(df.loc[idx_first_nan_power, 'timediff'] != '1s') & \
									(df.loc[idx_first_nan_power, 'power_shift'] == 0)]

	for i in idx_first_nan_power:
		print(df.loc[i-20:i+10, ['local_timestamp', 'power', 'file_id']])

	for f in df.file_id.unique():
		df_f = df[df.file_id == f]
		idx_zero_power = df_f[df_f.power == 0].index
		idx_nan_power = df_f[df_f.power == 0].index




	"""
	OLD: use this when we don't remove nans for local timestamps, or when we don't combine both timestamps
	nan_ts = df['timestamp'].notna()
	nan_lts = df['local_timestamp'].notna()
	nan_ltsl = df['local_timestamp_loc'].notna()

	print_times_dates("duplicated timestamps", 
		df[nan_ts], df[nan_ts]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts], df[nan_lts]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl], df[nan_ltsl]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')

	dupl_timestamp = df[nan_ts][df[nan_ts]['timestamp'].duplicated()]
	dupl_timestamp_first = df[nan_ts]['timestamp'].duplicated(keep='first')
	dupl_timestamp_last = df[nan_ts]['timestamp'].duplicated(keep='last')
	dupl_timestamps_keep = df[nan_ts][df[nan_ts]['timestamp'].duplicated(keep=False)].sort_values('timestamp')

	# TODO: find out a way to combine this info. 
	print("Are there glucose values in the duplicate data that is not dropped? ",
		not df[nan_ts][df[nan_ts]['timestamp'].duplicated(keep=False)].sort_values('timestamp').glucose.dropna().empty)
	print("Are there glucose values in the duplicate that that will be dropped? ",
		not df[nan_ts][df[nan_ts]['timestamp'].duplicated()].sort_values('timestamp').glucose.dropna().empty)

	dupl_devices = []
	for dev in devices:
		print("Duplicate timestamps in %s: "%dev, df[nan_ts][dupl_timestamp_first][dev].sum())
		if df[nan_ts][dupl_timestamp_first][dev].sum() != 0:
			dupl_devices.append(dev)
	# TODO: make sure that this happens in the right order, and that not the original duplicate is removed
	# TODO: find out why garmin is double and if data is the same then (latter: not)

	df['drop_dupl_timestamp'] = df[dupl_devices].sum(axis=1).astype(bool)
	"""

	"""
	print("\n----------------- SELECT: device_0 == ELEMNT")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & df['device_ELEMNTBOLT']], 
		df[nan_ts & df['device_ELEMNTBOLT']]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & df['device_ELEMNTBOLT']], 
		df[nan_lts & df['device_ELEMNTBOLT']]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & df['device_ELEMNTBOLT']], 
		df[nan_ltsl & df['device_ELEMNTBOLT']]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	# TODO: check if duplicated local timestamps were in error_timestamps_fileid

	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_ltsl | (df['device_ELEMNTBOLT'] | df['device_zwift'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	
	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift or Rouvy")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')

	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift or Rouvy or Garmin")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	"""

	"""
	df_dupl = df[df['timestamp'].duplicated(keep=False)]
	if not df_dupl.empty:
		df_dupl['date'] = df_dupl['timestamp'].dt.date
		df_dupl[df_dupl['timestamp'] == df_dupl.iloc[0]['timestamp']]

		print("Dates with multiple files but not duplicate timestamps:\n",
			*tuple(set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist()) 
				- set(df_dupl['date'].unique())))
		print("Dates with duplicate timestamps but not multiple files:\n",
			*tuple(set(df_dupl['date'].unique()) 
				- set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist())))
		# conclusion: duplicate timestamps must be because there are duplicate files
	"""
