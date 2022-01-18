# TODO: check if we indeed can use event subtype to fill extremes with
# TODO: watch out for filtering by Event Type!!
# TODO: preprocess glucose values (look online on internet)
# TODO: also look for sensor errors
import os
import gc

import numpy as np
import pandas as pd

from helper import *
from calc import *
from config import rider_mapping
from timezone import get_timezones_dexcom, get_timezones_final

from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

root = '/wave/hypex/data/Dexcom/'#'./data/Dexcom/'

if not os.path.exists(root+'drop/'):
    os.mkdir(root+'drop/')
if not os.path.exists(root+'clean/'):
    os.mkdir(root+'clean/')

def clean_export(df):
    """
    Clean individual export files: 
    - extract and correct patient info
    - drop alert info in first rows
    """
    # extract patient info of rider
    df_patient = df[(df['Event Type'] == 'FirstName') | (df['Event Type'] == 'LastName') | (df['Event Type'] == 'DateOfBirth')]

    # extract name and correct for name mistakes in dexcom CLARITY
    name = df_patient.loc[df['Event Type'] == 'LastName', 'Patient Info'].str.lower().replace({'declan':'irvine', 'clancey':'clancy'}).values[0]

    df = df.drop('Patient Info', axis=1)
    df = df.drop(df_patient.index)

    df['RIDER'] = name

    # drop alert info
    df = df[(df['Event Type'] != 'Device') & (df['Event Type'] != 'Alert')]

    return df

def merge_export(source):
    """
    Merge exports (90 days files) into one csv file
    """
    df = pd.concat([clean_export(pd.read_csv(root+'export/%s/%s'%(source,f))) for f in os.listdir(root+'export/%s/'%source)])

    # remove old index
    df = df.drop('Index', axis=1)
    df = df.reset_index(drop=True)

    # sort riders on order of rider mapping
    df = df.reset_index().sort_values(by=['RIDER', 'index'], key=lambda x: x.map(rider_mapping)).drop('index', axis=1)
    df = df.reset_index(drop=True)

    df.to_csv(root+'clean/TNN_CGM_2015-2021_%s_export_20211119.csv'%source)
    return df

def clean_glucose(df):
    """
    - remove "high" and "low" from event type and glucose values, and replace with 40 and 400
    - replace 0 with nan in glucose value column if the event type is not EGV or calibration
    - convert all mmol/L to mg/dL
    """
    # remove text "high" and "low" from glucose values and save in "EXTREME"
    unit = df.columns[df.columns.str.startswith('Glucose Value')].str.split()[0][-1]

    print("CLEAN zeros in Glucose Value if not EGV or calibration")
    df.loc[(df['Event Type'] != 'EGV') & (df['Event Type'] != 'Calibration'), f'Glucose Value {unit}'] = np.nan
    print("CHECK Are there remaining zero glucose values: ", not df[df[f'Glucose Value {unit}'] == 0].empty)

    print("MOVE 'high' and 'low' in Glucose Value to EXTREME")
    df[f'Glucose Value EXTREME'] = df[f'Glucose Value {unit}'].apply(lambda x: x if isinstance(x, str) else np.nan)
    df[f'Glucose Value {unit}'] = pd.to_numeric(df[f'Glucose Value {unit}'], errors='coerce')

    print("MOVE 'high' and 'low' in Event Subtype to EXTREME")
    mask = (df['Event Type'] == 'EGV') & df['Event Subtype'].notna()
    df.loc[mask, 'Glucose Value EXTREME'] = df.loc[mask, 'Glucose Value EXTREME'].fillna(df.loc[mask, 'Event Subtype']) 
    df.loc[mask, 'Event Subtype'] = np.nan

    if unit == '(mmol/L)':
        print("CONVERT mmol/L to mg/dL")
        df['Glucose Value (mg/dL)'] = df['Glucose Value (mmol/L)'] * mmoll_mgdl
        df = df.drop(f'Glucose Value {unit}', axis=1)

    print("REPLACE Low with 40 and High with 400 in EXTREME")
    print("FILLNA Glucose Value with EXTREME")
    df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].fillna(df['Glucose Value EXTREME'].replace({'Low':40., 'High':400.})).astype(float)
    df = df.drop('Glucose Value EXTREME', axis=1)
    
    return df

def clean_raw(source):
    """
    Read in the merged exports and perform a first clean:
    - anonymize data
    - add source column
    - rename timestamp
    - remove empty columns
    - remove "high" and "low" from event type and glucose values, and replace with 40 and 400
    - replace 0 with nan in glucose value column if the event type is not EGV or calibration
    - convert all mmol/L to mg/dL
    """
    df = pd.read_csv(root+'clean/TNN_CGM_2015-2021_%s_export_20211119.csv'%source, index_col=0)

    # anonymize file
    df.RIDER = df.RIDER.apply(lambda x: rider_mapping[x.lower()])

    # source
    df['source'] = 'Dexcom CLARITY '+source

    # timestamp
    df = df.rename({'Timestamp (YYYY-MM-DDThh:mm:ss)':'local_timestamp'}, axis=1)
    df.local_timestamp = pd.to_datetime(df.local_timestamp)

    # remove empty columns
    df = df.dropna(axis=1, how='all')

    # clean out glucose columns
    df = clean_glucose(df)

    return df

def fix_errors_manual_timezone(df):
    """
    manual device timezone mistakes
    """
    # first step is to assign transmitter id where it is missing
    df.loc[df['Transmitter ID'].isna() & (df['Event Type'] == 'EGV'), 'Transmitter ID'] = 'UNK_ID'

    # second step is to correct for mistakes by riders in manually switching timezones of their receiver
    # was setup with wrong date
    df.loc[(df.RIDER == 4)\
        & (df['Source Device ID'] == 'PL82609380')\
        & (df['Transmitter ID'] == '80CPYD')\
        & (df['Transmitter Time (Long Integer)'] <= 6248282), 'local_timestamp'] += pd.to_timedelta('-1days') 
    print("FIX (4) 80CPYD timestamps date wrong by 1 day")

    # was setup with wrong date
    df.loc[(df.RIDER == 4)\
        & (df['Source Device ID'] == 'PL82609380')\
        & (df['Transmitter ID'] == '810APT')\
        & (df['Transmitter Time (Long Integer)'] >= 4249412)\
        & (df['Transmitter Time (Long Integer)'] <= 4637304), 'local_timestamp'] += pd.to_timedelta('1days') 
    print("FIX (4) 810APT timestamps date wrong by 1 day")

    # changed it to the wrong month
    df.loc[(df.RIDER == 4)\
        & (df['Source Device ID'] == 'PL82609380')\
        & (df['Transmitter ID'] == '810APT')\
        & (df['Transmitter Time (Long Integer)'] >= 6233271)\
        & (df['Transmitter Time (Long Integer)'] <= 6879458), 'local_timestamp'] += pd.to_timedelta('30days 23:55:04') 
    print("FIX (4) 810APT timestamps october -> november")

    # was setup with wrong date
    # note: it's easier to do this in one step because of some transmitter confusion in the next step
    # and we CAN do this in one step, because max transmitter time (40M6JF) < min transmitter time (40M0TB)
    # for device SM64410763
    df.loc[(df.RIDER == 10)\
        & (df['Source Device ID'] == 'SM64410763')\
        & ((df['Transmitter ID'] == '40M0TB') | (df['Transmitter ID'] == '40M6JF'))\
        & ((df['Transmitter Time (Long Integer)'] >= 7963912) | (df['Transmitter Time (Long Integer)'] <= 104698)), 'local_timestamp'] += pd.to_timedelta('1days') 
    print("FIX (10) SM64410763 timestamps date wrong by 1 day")

    # had his device setup with the wrong year
    df.loc[(df.RIDER == 10)\
        & (df['Source Device ID'] == 'PL82501087')\
        & (df['local_timestamp'] <= '2018-01-22 17:40:57'), 'local_timestamp'] += pd.to_timedelta('365days') 
    print("FIX (10) PL82501087 timestamps 2018->2019")

    # reset the timestamp after travelling and switched the month and day around (10/4 instead of 4/10)
    df.loc[(df.RIDER == 15)\
        & (df['Source Device ID'] == 'PL82501061')\
        & (df['Transmitter ID'] == '809T66')\
        & (df['Transmitter Time (Long Integer)'] >= 5674414)\
        & (df['Transmitter Time (Long Integer)'] <= 6491302), 'local_timestamp'] += pd.to_timedelta('177days')
    print("FIX (15) PL82501061 timestamps from 10/4 to 4/10")

    # reset the timestamp after travelling and switched the month and day around (10/7 instead of 7/10)
    df.loc[(df.RIDER == 15)\
        & (df['Source Device ID'] == 'PL82501061')\
        & (df['Transmitter ID'] == '80YBT4')\
        & (df['Transmitter Time (Long Integer)'] >= 4163883)\
        & (df['Transmitter Time (Long Integer)'] <= 6179527), 'local_timestamp'] += pd.to_timedelta('90days')
    print("FIX (15) PL82501061 timestamps from 7/10 to 10/7")

    # was setup with wrong date
    df.loc[(df.RIDER == 15)\
        & (df['Source Device ID'] == 'PL82501061')\
        & (df['Transmitter ID'] == '80YBT4')\
        & (df['Transmitter Time (Long Integer)'] >= 6179827), 'local_timestamp'] += pd.to_timedelta('1days')
    print("FIX (15) PL82501061 timestamps date wrong by 1 day")
    df.loc[(df.RIDER == 15)\
        & (df['Source Device ID'] == 'PL82501061')\
        & (df['Transmitter ID'] == '8HLEHG'), 'local_timestamp'] += pd.to_timedelta('1days')
    print("FIX (15) PL82501061 timestamps date wrong by 1 day")

    # had his device setup with the wrong year 3 times
    # easiest fix (that also includes calibration errors) is to shift everything below 2016-06-13 06:55:20 to one year up
    df.loc[(df.RIDER == 18)\
        & (df['Source Device ID'] == 'SM64411240')\
        & (df['local_timestamp'] <= '2016-06-13 06:55:20'), 'local_timestamp'] += pd.to_timedelta('365days') 
    print("FIX (18) SM64411240 timestamps 2016->2017")

    # also the first dates were setup wrong
    df.loc[(df.RIDER == 18)\
        & (df['Source Device ID'] == 'SM64411240')\
        & (df['Transmitter ID'] == '40M63E')\
        & (df['Transmitter Time (Long Integer)'] <= 1652609), 'local_timestamp'] += pd.to_timedelta('89days') 
    print("FIX (18) SM64411240 timestamps 89 days")

    df.loc[(df.RIDER == 18)\
        & (df['Source Device ID'] == 'SM64411240')\
        & (df['Transmitter ID'] == '40M63E')\
        & (df['Transmitter Time (Long Integer)'] <= 249887), 'local_timestamp'] += pd.to_timedelta('-1days') 
    print("FIX (18) SM64411240 timestamps 1 days")

    df = df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'])
    df = df.reset_index(drop=True)
    return df

def fix_errors_transmitter_id(df):
    # --------------------- error dexcom transmitter ID
    # OBSERVATION We see the following:
    # - There is a time window when both the first and second transmitter are observed, alternating a bit
    # - In the time that we observed both the first and second transmitter, 
    #   the transmitter time of the first transmitter continues. 
    # - The transmitter time of the second transmitter is at some point reset to zero (or 7500)

    # CONCLUSION Therefore we conclude that the riders probably continued using the first transmitter
    # (longer than they should have?) and this messed up the system.

    # SOLUTION The solution is to change the transmitter ID in the period that we observe both the old
    # and the new transmitter, to only the ID of the old transmitter. Then all issues should be fixed.

    df.loc[(df.RIDER == 4) & (df.local_timestamp >= '2019-07-13 05:11:07')\
        & (df['Transmitter ID'] == '80CW29'), 'Transmitter ID'] = 'UNK_ID'
    print("FIX (4) transmitter ID between 2019-07-13 05:11:07 and 2019-08-22 05:24:16 from 80CW29 to UNK_ID")

    df.loc[(df.RIDER == 6) & (df.local_timestamp <= '2019-03-27 05:24:16')\
        & (df['Transmitter ID'] == '80QJ2F'), 'Transmitter ID'] = '80LF01'
    print("FIX (6) transmitter ID between 2019-03-05 10:50:39 and 2019-03-27 05:24:16 from 80QJ2F to 80LF01")

    df.loc[(df.RIDER == 6) & (df.local_timestamp <= '2018-11-25 15:29:29')\
        & (df['Transmitter ID'] == '80LF01'), 'Transmitter ID'] = '80CPX2'
    print("FIX (6) transmitter ID between 2018-09-12 07:47:32 and 2018-11-25 15:29:29 from 80LF01 to 80CPX2")
    # TODO: he synched with two devices at the same time, so if there are duplicates from the transmitter time, remove them

    df.loc[(df.RIDER == 6) & (df.local_timestamp <= '2019-09-24 20:59:16')\
        & (df['Transmitter ID'] == '80RE8H'), 'Transmitter ID'] = '80UKML'
    print("FIX (6) transmitter ID between 2019-08-22 21:25:41 and 2019-09-24 20:59:16 from 80RE8H to 80UKML")

    df.loc[(df.RIDER == 10) & (df.local_timestamp <= '2017-03-18 20:22:40')\
        & (df['Transmitter ID'] == '40M6JF'), 'Transmitter ID'] = '40M0TB'
    print("FIX (10) transmitter ID between 2017-03-11 02:13:23 and 2017-03-18 20:22:40 from 40M6JF to 40M0TB")
    # TODO: he synched with two devices at the same time, so if there are duplicates from the transmitter time, remove them

    df.loc[(df.RIDER == 14) & (df.local_timestamp <= '2020-02-06 13:24:30')\
        & (df['Transmitter ID'] == '8JJ0MQ'), 'Transmitter ID'] = '810C8M'
    print("FIX (14) transmitter ID between 2020-01-27 16:40:03 and 2020-02-06 13:24:30 from 8JJ0MQ to 810C8M")

    df.loc[(df.RIDER == 14) & (df.local_timestamp <= '2019-01-18 22:36:15')\
        & (df['Transmitter ID'] == '80RRBL'), 'Transmitter ID'] = '80JPC8'
    print("FIX (14) transmitter ID between 2019-01-17 15:56:19 and 2019-01-18 22:36:15 from 80RRBL to 80JPC8")

    df.loc[(df.RIDER == 15) & (df.local_timestamp <= '2019-08-20 13:51:12')\
        & (df['Transmitter ID'] == 'UNK_ID'), 'Transmitter ID'] = '80RNWS'
    print("FIX (15) transmitter ID between 2019-07-21 14:27:36 and 2019-08-20 13:51:12 from UNK_ID to 80RNWS")

    df.loc[(df.RIDER == 15) & (df.local_timestamp >= '2019-08-21 08:17:53')\
        & (df['Transmitter ID'] == 'UNK_ID'), 'Transmitter ID'] = '80YBT4'
    print("FIX (15) transmitter ID between 2019-08-21 08:17:53 and 2019-09-15 16:46:21 from UNK_ID to 80YBT4")

    df.loc[(df.RIDER == 17) & (df.local_timestamp <= '2018-09-14 09:12:34')\
        & (df['Transmitter ID'] == '80UK8Y'), 'Transmitter ID'] = '80CU6B'
    print("FIX (17) transmitter ID between 2018-09-06 20:07:52 and 2018-09-14 09:12:34 from 80UK8Y to 80CU6B")

    df.loc[(df.RIDER == 18) & (df.local_timestamp <= '2019-06-22 08:32:45')\
        & (df['Transmitter ID'] == '8GM9KD'), 'Transmitter ID'] = '80D24X'
    print("FIX (18) transmitter ID between 2019-06-22 08:02:45 and 2019-06-22 08:32:45 from 8GM9KD to 80D24X")

    df = df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)', 'source'])
    df = df.reset_index(drop=True)

    return df

def select_date_range(df, d_min, d_max):
    # select by date range
    df = df[(df.local_timestamp.dt.date <= d_max) & (df.local_timestamp.dt.date >= d_min)]
    print(f"DROPPED entries after {d_max} or before {d_min}")

    df = df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)', 'source'])
    df = df.reset_index(drop=True)
    return df

def drop(df, dname, mask):
    df[mask].to_csv(root+'drop/'+dname+'.csv')
    print("DROP %s "%mask.sum()+dname)
    return df[~mask]

def drop_duplicates_nans(df):
    # ------- duplicates
    # drop duplicates rows
    df = drop(df, 'duplicated_rows',
        df.drop('source', axis=1).duplicated(keep='first'))

    # drop duplicates rows where glucose value is not exactly the same, but the rest is
    # this mostly occurs when data is both in eu and us, and in the merge, 
    # the glucose value from unit conversion (mmol/L to mg/dL) is not exactly the same
    # keep the ones that are from the US
    df = drop(df, 'duplicated_rows_noglucose',
        df.drop(['source', 'Glucose Value (mg/dL)'], axis=1).duplicated(keep='last'))

    df = df.sort_values(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)', 'source', 'Source Device ID'])
    # recording with two devices at the same time and data downloaded from the same source (CLARITY EU/US)
    df = drop(df, 'duplicated_rows_fromtworeceivers',
        (df['Event Type'] == 'EGV') & df.duplicated(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)', 'source'], keep='first'))

    # recording with two devices at the same time and data downloaded from a different source (CLARITY EU/US)
    df = drop(df, 'duplicated_rows_fromtworeceivers_differentsource',
        (df['Event Type'] == 'EGV') & df.duplicated(['RIDER', 'Transmitter ID', 'Transmitter Time (Long Integer)'], keep='last'))

    df = df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'])
    df = df.reset_index(drop=True)

    # ------- nans
    df = drop(df, 'nan_rows', df[['Insulin Value (u)', 'Carb Value (grams)', 'Duration (hh:mm:ss)', 'Glucose Value (mg/dL)']].isna().all(axis=1)
                & (df['Event Type'] != 'Insulin') & (df['Event Type'] != 'Health'))

    return df

def sort_transmitter_time(df):
    # ------- transmitter check
    # check if there are any readings that are not EGV or Calibration and that do have a transmitter ID
    print("CHECK Number of readings that are not EGV or Calibration and that do have a transmitter ID: ", 
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
    for i in df.RIDER.unique():
        df_i = df_transmitter.loc[i]
        for j in range(len(df_i)-1):
            if df_i.iloc[j][('local_timestamp', 'max')] >= df_i.iloc[j+1][('local_timestamp', 'min')]:
                print(i, df_i.iloc[j:j+2])

    """
    Overlap transmitters
    10                    local_timestamp                    
                                   min                 max
    Transmitter ID                                        
    419RDG         2018-07-21 03:00:20 2018-08-02 09:34:25
    809W41         2018-07-31 23:34:08 2018-11-10 23:14:45
    potentially during the transmitter 809W41, the timestamps were 3 days back
    """

    # --------------------- sort by transmitter time
    # Create transmitter order
    transmitter_order = {df_transmitter.reset_index()['Transmitter ID'][n]:n for n in np.arange(len(df_transmitter))}
    df['transmitter_order'] = df['Transmitter ID'].apply(lambda x: transmitter_order[x] if x in transmitter_order.keys() else len(transmitter_order))
    del transmitter_order ; gc.collect()

    # Split in EGV and non-EGV for sorting
    df_egv = df[df['Event Type'] == 'EGV']
    df_nonegv = df[df['Event Type'] != 'EGV']

    # Sort by: Event Type - RIDER - transmitter_order - Transmitter Time
    df_egv = df_egv.sort_values(by=['RIDER', 'transmitter_order', 'Transmitter Time (Long Integer)', 'Source Device ID'])
    df = df_egv.append(df_nonegv)
    df = df.reset_index(drop=True)

    # For each non-EGV reading, put it in the right rider + time window
    for idx, (i, t) in tqdm(df.loc[df['Event Type'] != 'EGV', ['RIDER', 'local_timestamp']].iterrows()):
        loc = df.index.get_loc(idx)

        # TODO: what if during travelling?
        prev_df = df[(df.RIDER == i) & (df.local_timestamp < t) & (df['Event Type'] == 'EGV')]
        if not prev_df.empty:
            idx_new = prev_df.index[-1]
            loc_new = df.index.get_loc(idx_new)

            df = df.loc[np.insert(np.delete(df.index, loc), loc_new+1, loc)]
    df = df.reset_index(drop=True)
    return df

def time_to_utc(df, df_changes):
    # convert incorrect local timestamps to UTC time
    for (i,n), (idx_min, idx_max, _, _, tz) in df_changes.iterrows():
        df.loc[idx_min:idx_max, 'timestamp'] = df.loc[idx_min:idx_max, 'local_timestamp'] - tz

    df = df.sort_values(['RIDER', 'timestamp'])
    df = df.reset_index(drop=True)
    return df

def check_time(df):
    # check if it worked
    df_egv = df[df['Event Type'] == 'EGV']
    df_egv['timestamp_diff'] = df_egv['timestamp'].diff()
    df_egv['transmitter_diff'] = df_egv['Transmitter Time (Long Integer)'].diff()

    df_egv['timediff'] = df_egv['timestamp_diff'] - pd.to_timedelta(df_egv['transmitter_diff'], 'sec')
    df_egv.loc[df_egv['transmitter_order'].diff() != 0, 'timediff'] = np.nan # correct for transmitter change

    df_egv['change'] = (df_egv['timediff'] < '-5min') | (df_egv['timediff'] > '5min')

    print("Number of gaps left: ", (df_egv['timediff'] > '5min').sum())
    print("Number of dups left: ", (df_egv['timediff'] < '-5min').sum())
    print("Number of changes left: ", df_egv['change'].sum())

    print("When transmitter time goes down: ",
        df_egv.loc[(df_egv['transmitter_diff'] < 0) & (df_egv.RIDER.diff() == 0) & (df_egv['transmitter_order'].diff() == 0) \
        & (df_egv['Event Type'].shift() == df_egv['Event Type']) & (df_egv['Event Type'] == 'EGV'),
        ['RIDER', 'Event Type', 'local_timestamp', 'Source Device ID', 'Transmitter ID', 'Transmitter Time (Long Integer)', 
        'source', 'transmitter_order', 'timestamp', 'timestamp_diff', 'transmitter_diff']])
    # seems to happen mostly when there are two receivers
    """
    for i in df['RIDER'].unique():
        plot_time(df, i, x='timestamp')

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

def utc_to_localtime(df, tz):
    tz['n'] = tz.groupby('RIDER')['timezone'].transform(lambda x: (x.shift() != x).cumsum())
    tz_gb = tz.groupby(['RIDER', 'n']).agg({'timezone' :'first', 'date':['min', 'max']})
    tz_gb.columns = [i[0]+'_'+i[1] if i[0] == 'date' else i[0] for i in tz_gb.columns]

    # recalculate local timestamp
    df = df.rename(columns={'local_timestamp':'local_timestamp_raw'})

    for (i, _), (tz, date_min, date_max) in tz_gb.iterrows():
        mask_tz = (df.RIDER == i) & (df.timestamp.dt.date >= date_min) & (df.timestamp.dt.date <= date_max)
        df.loc[mask_tz, 'local_timestamp'] = df.loc[mask_tz, 'timestamp'] + tz

    return df

def plot_time(df, i, x='local_timestamp', y='Transmitter Time (Long Integer)', hue='Transmitter ID', save_to=True):
    df_i = df[df.RIDER == i]
    plt.figure(figsize=(20,5))
    sns.lineplot(data=df_i, x=x, y=y, hue=hue)
    if save_to:
        plt.savefig(f'{root}/{i}_{x}_transmittertime.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_time_transmitter(df, i, tid, x='local_timestamp', y='Transmitter Time (Long Integer)', save_to=True):
    df_t = df[(df.RIDER == i) & (df['Transmitter ID'] == tid)]
    plt.plot(df_t[x], df_t[y])
    if save_to:
        plt.savefig(f'{root}/{i}_{tid}_{x}_transmittertime.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def main():
    df_eu = merge_export('EU')
    df_us = merge_export('US')

    df_eu = clean_raw('EU')
    df_us = clean_raw('US')

    # merge EU and US
    df = pd.merge(df_eu, df_us, how='outer',
        on=df_us.columns.drop(['source', 'Glucose Value (mg/dL)']).tolist())

    # if item appears in both US and EU, keep the ones that appear in the US dataframe
    df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)_y'].fillna(df['Glucose Value (mg/dL)_x'])
    df['source'] = df['source_y'].fillna(df['source_x'])
    df = df.drop(['Glucose Value (mg/dL)_x', 'Glucose Value (mg/dL)_y', 'source_x', 'source_y'], axis=1)

    # sort and save
    df = df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'])
    df = df.reset_index(drop=True)
    df.to_csv(root+'clean/dexcom_raw.csv')

    # ------- fix errors
    df = fix_errors_manual_timezone(df)
    df = fix_errors_transmitter_id(df)

    """
    # CHECK timestamp and transmitter correction
    for i in df['RIDER'].unique():
        plot_time(df, i)

    for _, (i, tid) in df[['RIDER', 'Transmitter ID']].drop_duplicates().iterrows():
        print(i, tid)
        plot_time_transmitter(df, i, tid)
    """
    # select by date
    #df = select_date_range(df, d_min=datetime.date(2018,12,1), d_max=datetime.date(2019,11,30))

    # drop duplicates and nans
    df = drop_duplicates_nans(df)

    # sort by rider, transmitter, transmitter_time
    df = sort_transmitter_time(df)

    df.to_csv(root+'clean/dexcom_sorted.csv')

    ################ PREREQUISITE: TrainingPeaks timezone_final_list.csv
    """
    df = pd.read_csv(root+'clean/dexcom_sorted.csv', index_col=0)
    df.local_timestamp = pd.to_datetime(df.local_timestamp)
    """

    # get list with timezone changes
    df_changes = get_timezones_dexcom(df)
    df_changes.to_csv(root+'clean/timezone_dexcom.csv')

    # convert incorrect local time to utc
    df = time_to_utc(df, df_changes)
    check_time(df)
    df = df.drop('transmitter_order', axis=1)
    df.to_csv(root+'clean/dexcom_utc.csv')

    tz = get_timezones_final(df)
    tz.to_csv(root.rstrip('Dexcom/')+'/timezone.csv')

    # TODO: fix all insulin and carbs metrics
    df = utc_to_localtime(df, tz)
    df.to_csv(root+'clean/dexcom_clean.csv')

if __name__ == '__main__':
    main()