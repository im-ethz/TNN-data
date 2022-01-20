import pandas as pd

# glucose levels in mg/dL
glucose_levels = {'hypo L2': (0,53),
				  'hypo L1': (54,69),
				  'target' : (70,180),
				  'hyper L1': (181,250),
				  'hyper L2': (251,10000)}

mmoll_mgdl = 18
mgdl_mmoll = 1/mmoll_mgdl

def semicircles_to_degrees(df:pd.Series):
	# convert latitude and longitude from semicircles to degrees
	return df * (180 / 2**31)

def degrees_to_semicircles(df:pd.Series):
	# convert latitude and longitude from semicircles to degrees
	return df / (180 / 2**31)

def elevation_gain(altitude: pd.Series):
	# calculate the total elevation gain during a workout
	return altitude.diff()[altitude.diff() > 0].sum()

def elevation_loss(altitude: pd.Series):
	# calculate the total elevation loss during a workout
	return altitude.diff()[altitude.diff() < 0].sum()