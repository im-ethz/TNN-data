import os
import sys
sys.path.append(os.path.abspath('../../../'))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from pandas_profiling import ProfileReport

import datetime
import geopy
import pytz
from pytz import country_names, country_timezones
country_names_inv = {v:k for k,v in country_names.items()}

from tzwhere import tzwhere
tzwhere = tzwhere.tzwhere()

from config import rider_mapping
rider_mapping_inv = {v:k for k, v in rider_mapping.items()}

from plot import *
from helper import *

import gc

# fourth stage: aggregation
# a) aggregate by training session
# b) aggregate by day

fitness = pd.read_csv('../../fitness/fitness_analysis_2019_anonymous.csv')

df_agg = []

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'clean3/')])

for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(path+'clean3/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

# stats:
# mean, std, min, max, np.ptp, sum

